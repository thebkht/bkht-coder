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
DEFAULT_NUM_CTX = 32768


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
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[Chunk]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": self.num_ctx},
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
