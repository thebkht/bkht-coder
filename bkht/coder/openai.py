"""The OpenAI-compatible backend.

One wire format, four servers. ``mlx_lm.server``, llama.cpp's ``llama-server``,
vLLM and Ollama's own ``/v1`` all speak ``POST /v1/chat/completions``, so a
single provider covers every way a model gets served on a machine somebody
owns -- including the fine-tune this project trains, which Ollama cannot serve
without first being taught about it.

That is what makes this the default rather than an alternative. The promise has
never been "Ollama"; it is that the weights are on hardware you control. Point
``host`` at ``localhost`` and the promise holds exactly as before. Point it at
another machine on the network and it still holds -- which is the arrangement
this backend exists for: the Mac with the memory serves, and the laptop, the
tablet or the phone drives.

Three things differ from :class:`~bkht.coder.provider.OllamaProvider`, and each
is a place a naive port would break:

* The stream is SSE. Every payload arrives behind ``data: ``, and the stream is
  ended by a literal ``data: [DONE]`` rather than by a JSON object saying so.
* Native tool calls arrive in pieces. The ``arguments`` string is split across
  as many deltas as the server feels like, keyed by ``index``, so a call is only
  complete at the end of the stream.
* ``num_ctx`` is not something a request can ask for. The window is fixed when
  the server is started, so the number here is carried for context accounting
  and nothing else -- unlike Ollama, where sending it is load-bearing. If it and
  the server disagree, the server wins and coder's meter is the thing that is
  wrong; ``coder doctor`` says so.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx

from .parsing import ToolCall
from .provider import (
    CONNECT_TIMEOUT,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    OPENAI_HOST,
    OPENAI_MODEL,
    READ_TIMEOUT,
    Chunk,
    ProviderError,
)

#: Read from the environment rather than the config file, because a config file
#: is a thing people commit. Most local servers want no key at all, and the ones
#: that do accept any non-empty string.
API_KEY_ENV = "CODER_API_KEY"

#: What every SSE payload is prefixed with, and the sentinel that ends a stream.
DATA_PREFIX = "data:"
DONE = "[DONE]"


class OpenAIProvider:
    """Streams completions from any OpenAI-compatible server."""

    def __init__(
        self,
        model: str = OPENAI_MODEL,
        host: str = OPENAI_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = READ_TIMEOUT,
        temperature: float | None = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
        vision: bool = False,
    ) -> None:
        self.model = model
        self.host = (host or OPENAI_HOST).rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        # Not asked of the server: `/v1/models` reports ids and owners and says
        # nothing about what a model can take, and there is no `/api/show` here
        # to ask instead. False is the honest default -- the caller's next move
        # is to tell the user the picture will not be looked at, and for a
        # text-only local model that is true.
        self.vision = vision
        self.timeout = httpx.Timeout(
            timeout, connect=CONNECT_TIMEOUT, write=CONNECT_TIMEOUT
        )

    def can_see(self) -> bool:
        """Whether images are worth attaching; see ``vision``."""
        return self.vision

    # --- requests -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[Chunk]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_convert(message, self.vision) for message in messages],
            "stream": True,
            # Without this the usage block is omitted from a streamed response
            # entirely, and the context meter has nothing to count with.
            "stream_options": {"include_usage": True},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools

        # Deltas assemble here rather than in the caller: a tool call is not a
        # thing until the stream ends, so it is emitted once, on the last chunk.
        pending: dict[int, dict] = {}

        try:
            with httpx.stream(
                "POST",
                f"{self.host}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise ProviderError(
                        f"{self.host} returned {response.status_code}: "
                        f"{response.text.strip()[:400]}"
                    )
                for line in response.iter_lines():
                    chunk = self._parse_line(line, pending)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"cannot reach the model server at {self.host}: {exc}"
            ) from exc

        # The sentinel that ended the stream carries nothing, so the assembled
        # calls are emitted after it rather than on it.
        if pending:
            yield Chunk(done=True, tool_calls=_assemble(pending))

    def _parse_line(self, line: str, pending: dict[int, dict]) -> Chunk | None:
        """One SSE line as a chunk, or None for a line that says nothing.

        Blank lines separate events, and a line beginning with a colon is a
        keep-alive comment. Both are frequent and neither means anything.
        """
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if not line.startswith(DATA_PREFIX):
            return None
        body = line[len(DATA_PREFIX):].strip()
        if body == DONE:
            return None

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None

        # Some servers report a mid-stream failure as an ordinary event rather
        # than by closing with a status, so it has to be caught in both places.
        if error := data.get("error"):
            message = error.get("message") if isinstance(error, dict) else error
            raise ProviderError(str(message))

        chunk = Chunk()
        if usage := data.get("usage"):
            chunk.prompt_tokens = usage.get("prompt_tokens")
            chunk.completion_tokens = usage.get("completion_tokens")

        choices = data.get("choices") or []
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            chunk.content = delta.get("content") or ""
            _accumulate(delta.get("tool_calls"), pending)
            chunk.done = bool(choice.get("finish_reason"))

        return chunk

    def available(self) -> bool:
        """Whether the server answers, used to skip live tests."""
        try:
            httpx.get(
                f"{self.host}/v1/models", headers=self._headers(), timeout=2.0
            ).raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def deterministic(self) -> "OpenAIProvider":
        """A copy of this provider that samples at zero; see ``for_review``."""
        return OpenAIProvider(
            model=self.model,
            host=self.host,
            num_ctx=self.num_ctx,
            temperature=0.0,
            api_key=self.api_key,
            vision=self.vision,
        )

    def models(self) -> list[str] | None:
        """Every model id the server is serving, or None if it did not answer.

        None and an empty list mean different things -- no server, versus a
        server with nothing loaded -- and `coder doctor` reports them
        differently, so they are not collapsed here.
        """
        try:
            response = httpx.get(
                f"{self.host}/v1/models", headers=self._headers(), timeout=CONNECT_TIMEOUT
            )
            if response.status_code != 200:
                return None
            entries = response.json().get("data") or []
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        return [str(entry.get("id")) for entry in entries if isinstance(entry, dict)]


def _accumulate(deltas, pending: dict[int, dict]) -> None:
    """Fold one delta's worth of tool call into what has arrived so far.

    The name comes whole, on the first delta for an index; the arguments arrive
    as a JSON *string* in as many pieces as the server chose to send. Appending
    rather than replacing is the entire point -- a `+` mistaken for an `=` here
    leaves every call with only its last few characters, which parses as an
    empty object and fails silently.
    """
    for delta in deltas or []:
        if not isinstance(delta, dict):
            continue
        index = delta.get("index", 0)
        entry = pending.setdefault(index, {"name": "", "arguments": ""})
        function = delta.get("function") or {}
        if name := function.get("name"):
            entry["name"] = name
        if (arguments := function.get("arguments")) is not None:
            entry["arguments"] += arguments


def _assemble(pending: dict[int, dict]) -> list[ToolCall]:
    """The accumulated deltas as tool calls, in the order the server sent them.

    A call whose arguments never became valid JSON is dropped rather than passed
    on as an empty one: the agent loop answers a malformed call with a
    correction the model can act on, and a call that arrives looking well-formed
    but empty would get no such correction.
    """
    calls = []
    for index in sorted(pending):
        entry = pending[index]
        if not entry["name"]:
            continue
        try:
            arguments = json.loads(entry["arguments"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        calls.append(ToolCall(name=entry["name"], arguments=arguments, raw=dict(entry)))
    return calls


def _convert(message: dict, vision: bool) -> dict:
    """One coder message as this wire format takes it.

    The ``tool`` role is kept rather than flattened into a user turn. Qwen's
    chat template renders it correctly, and keeping it means the bytes a
    fine-tune is *trained* on and the bytes it is *served* are assembled by the
    same template -- which is the one alignment worth protecting, since a
    mismatch there is invisible until the model starts emitting calls coder
    cannot parse.

    Images become content parts. Dropped when the model cannot see, because a
    server that does not expect the list form will reject the whole request,
    and losing the picture is better than losing the turn.
    """
    paths = message.get("images")
    converted = {key: value for key, value in message.items() if key != "images"}
    if not paths or not vision:
        return converted

    from .clipboard import encode

    parts: list[dict] = []
    if text := converted.get("content"):
        parts.append({"type": "text", "text": text})
    for path in paths:
        try:
            data = encode(path)
        except OSError:
            continue
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}
        )
    converted["content"] = parts
    return converted
