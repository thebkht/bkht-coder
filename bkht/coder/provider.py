"""Model transport.

``Provider`` is a Protocol with a single implementation, ``OllamaProvider``.
The indirection is deliberate: adding a hosted backend later should be a new
file rather than a refactor.

Two things here are load-bearing for a local model:

* ``options.num_ctx`` is set explicitly. Ollama defaults to 2048 and silently
  truncates everything past it, which is the most common cause of a bad
  local-model experience.
* Streaming is required. Waiting 30 seconds with no output is the main thing
  that makes a local agent feel dead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol

import httpx

from .parsing import ToolCall, parse_tool_calls, strip_json

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
# Measured on a 16 GB M-series machine with qwen2.5-coder:14b (Q4, ~9 GB of
# weights), one trivial completion, warm:
#
#     num_ctx  placement                size   warm turn
#      8192    100% GPU                 10 GB      0.9 s
#     16384    9% CPU / 91% GPU         12 GB     11.1 s
#     32768    27% CPU / 73% GPU        15 GB    >300 s (timed out)
#
# 8192 is the fastest number in that table and the wrong default. The table
# measures one trivial completion; a real turn is a conversation, and at 8192 it
# cannot hold a source file and think at the same time -- this project's own
# cli.py is ~6,900 tokens, 85% of the window. The turn does not fail loudly. It
# reads a file, frees context to make room, loses the file, and reads it again,
# spending its whole iteration budget paging. Measured on the task that exposed
# this: 25 iterations and no answer at 8192, eight tool calls and a complete one
# at 16384.
#
# So the default pays 11 seconds a turn to be able to finish. 32768 is still out
# of reach -- the model advertises it, but past 16384 the KV cache pushes the
# working set off the GPU far enough that turns time out. On a machine with less
# memory, drop to 8192 with --num-ctx; `coder doctor` says when to.
DEFAULT_NUM_CTX = 16384

# Ollama's own default is 2048, which silently truncates instead of erroring.
# Anything at or below it is a misconfiguration rather than a small window.
MIN_USEFUL_NUM_CTX = 4096

# Ollama's default is 0.8. That is a reasonable setting for prose and a poor one
# here: every tool call is a JSON object that has to be exactly right, and the
# characteristic failure of a 14b model on this loop is drifting off the
# emission format. Low, not zero -- at 0.0 a model that has taken a wrong turn
# repeats it verbatim on every retry, and the retry exists to get a different
# answer.
DEFAULT_TEMPERATURE = 0.2

# Ollama unloads a model after five idle minutes by default. A turn that has to
# reload 9 GB of weights before its first token spends longer waiting than
# thinking, and between two turns of a conversation that is exactly what
# happens.
DEFAULT_KEEP_ALIVE = "30m"

# A dead server must fail in seconds; a loaded 14b legitimately takes minutes to
# produce its first token, so connect and read are bounded separately.
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 300.0


class ProviderError(RuntimeError):
    """The model could not be reached or returned an unusable response."""


@dataclass
class Chunk:
    """One streamed fragment of a reply."""

    content: str = ""
    done: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class Reply:
    """A complete reply, assembled from the stream.

    ``tool_calls`` merges both transports: calls parsed out of message content
    (the path that demonstrably works with qwen2.5-coder) and calls found in
    the native ``tool_calls`` field. The agent loop never knows which was used.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def prose(self) -> str:
        """The reply text with any extracted tool-call JSON removed."""
        return strip_json(self.content)


class Provider(Protocol):
    """Anything that can turn a message history into a streamed reply."""

    model: str

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[Chunk]: ...


def collect(chunks: Iterable[Chunk]) -> Reply:
    """Drain a chunk stream into a single :class:`Reply`."""
    reply = Reply()
    native_calls: list[ToolCall] = []

    for chunk in chunks:
        reply.content += chunk.content
        native_calls.extend(chunk.tool_calls)
        if chunk.prompt_tokens is not None:
            reply.prompt_tokens = chunk.prompt_tokens
        if chunk.completion_tokens is not None:
            reply.completion_tokens = chunk.completion_tokens

    # Content parsing is the primary transport; native calls are additive.
    reply.tool_calls = parse_tool_calls(reply.content) + native_calls
    return reply


def _native_tool_calls(message: dict) -> list[ToolCall]:
    """Normalize Ollama's ``message.tool_calls`` into our own type."""
    calls = []
    for entry in message.get("tool_calls") or []:
        function = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(
            ToolCall(name=name, arguments=arguments or {}, raw=dict(entry))
        )
    return calls


class OllamaProvider:
    """Streams completions from a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = READ_TIMEOUT,
        temperature: float | None = DEFAULT_TEMPERATURE,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        if num_ctx < MIN_USEFUL_NUM_CTX:
            raise ValueError(
                f"num_ctx of {num_ctx} is too small to be useful; Ollama's own "
                f"default of 2048 silently truncates the prompt. Use at least "
                f"{MIN_USEFUL_NUM_CTX}."
            )
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.timeout = httpx.Timeout(
            timeout, connect=CONNECT_TIMEOUT, write=CONNECT_TIMEOUT
        )

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[Chunk]:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools

        try:
            with httpx.stream(
                "POST",
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise ProviderError(
                        f"Ollama returned {response.status_code}: "
                        f"{response.text.strip()[:400]}"
                    )
                for line in response.iter_lines():
                    chunk = self._parse_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot reach Ollama at {self.host}: {exc}") from exc

    def _parse_line(self, line: str) -> Chunk | None:
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if error := data.get("error"):
            raise ProviderError(str(error))

        message = data.get("message") or {}
        return Chunk(
            content=message.get("content") or "",
            done=bool(data.get("done")),
            tool_calls=_native_tool_calls(message),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def available(self) -> bool:
        """Whether the server answers, used to skip live tests."""
        try:
            httpx.get(f"{self.host}/api/tags", timeout=2.0).raise_for_status()
        except httpx.HTTPError:
            return False
        return True


def for_review(provider: Provider) -> Provider:
    """A copy of ``provider`` that samples deterministically.

    Review is a measurement, not a conversation. At the default temperature the
    same diff yields findings on one run and an empty array on the next, which
    makes recall and precision unusable as a metric -- a prompt change and a
    dice roll look identical. Anything that is not an OllamaProvider is
    returned unchanged, so fakes and future backends still work.
    """
    if not isinstance(provider, OllamaProvider):
        return provider
    return OllamaProvider(
        model=provider.model,
        host=provider.host,
        num_ctx=provider.num_ctx,
        temperature=0.0,
        keep_alive=provider.keep_alive,
    )
