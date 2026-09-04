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

#: The backend a session runs on unless it is told otherwise. Local, because
#: that is the promise this project makes; the others are opt-in.
#:
#: An OpenAI-compatible server rather than Ollama, because a fine-tune trained
#: on this machine is served by `mlx_lm.server` and Ollama cannot host it
#: without being taught about it first. The promise is unchanged: `local`
#: points at localhost by default, and the weights stay on hardware the user
#: owns. Ollama remains fully supported -- it stopped being the default, not
#: the recommendation for anyone running a stock model.
DEFAULT_PROVIDER = "local"

#: Ollama's own defaults. Named for it specifically, because they are no longer
#: the defaults of the program -- `DEFAULTS` below decides that per backend.
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

#: Where `mlx_lm.server` and llama.cpp's `llama-server` both listen, and the
#: name a server hosting exactly one model ignores. Declared here rather than
#: in ``openai`` so that ``DEFAULTS`` can name them without importing the
#: module -- naming a backend is what loads it, and that is worth keeping.
OPENAI_HOST = "http://localhost:8080"
OPENAI_MODEL = "coder"

# Ollama's default is 0.8. That is a reasonable setting for prose and a poor one
# here: every tool call is a JSON object that has to be exactly right, and the
# characteristic failure of a 14b model on this loop is drifting off the
# emission format. Low, not zero -- at 0.0 a model that has taken a wrong turn
# repeats it verbatim on every retry, and the retry exists to get a different
# answer.
DEFAULT_TEMPERATURE = 0.2

# The two agent CLIs coder can borrow a model from, and the window each one
# reports. Unlike `num_ctx` on Ollama these are not something a request asks
# for -- the window is a property of the model, and coder only needs the number
# to know when a history is getting close to it. Set `num_ctx` lower to have
# long sessions compact themselves sooner.
DEFAULT_CLAUDE_CODE_MODEL = "opus"
CLAUDE_CODE_NUM_CTX = 1_000_000

DEFAULT_CODEX_MODEL = "gpt-5.5"
CODEX_NUM_CTX = 400_000

# Ollama unloads a model after five idle minutes by default. A turn that has to
# reload 9 GB of weights before its first token spends longer waiting than
# thinking, and between two turns of a conversation that is exactly what
# happens.
DEFAULT_KEEP_ALIVE = "30m"

# A dead server must fail in seconds; a loaded 14b legitimately takes minutes to
# produce its first token, so connect and read are bounded separately.
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 300.0


#: The window above which the small-model adaptations stop applying.
#:
#: Several behaviours in this program exist because a 14b model at 16384 cannot
#: hold what it read: one tool call per reply, refusing a byte-identical
#: repeat, searching the workspace before the model has even seen the request.
#: Each was measured, each is right at 16384, and each is wrong at 400,000 --
#: where a second read of a file is ordinary rather than a symptom, and a turn
#: that batches four reads is a turn that finishes.
#:
#: 65536 because nothing shipped here is near it -- the local default is 16384
#: -- and nothing a large backend reports is below it: 400,000 for `codex`,
#: 1,000,000 for `claude-code`. A local server configured past it is making the
#: same claim those do, and is taken at its word.
ROOMY_NUM_CTX = 65_536


def roomy_window(num_ctx: int) -> bool:
    """Whether ``num_ctx`` is large enough to drop the small-model rules.

    Takes the number rather than the provider because one caller has only the
    number: `config` settles the scout before a provider has been built.
    """
    return (num_ctx or 0) >= ROOMY_NUM_CTX


def roomy(provider) -> bool:
    """Whether ``provider`` has room enough to drop the small-model rules.

    Asked of the window rather than of the backend name. The window is the
    thing the adaptations are actually about, it is already reported by every
    provider here, and it is a number the user can set -- so somebody running a
    128k local model gets the same treatment as a frontier one without this
    file needing to have heard of their model.
    """
    return roomy_window(getattr(provider, "num_ctx", 0))


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


def _with_images(message: dict) -> dict:
    """A message with its image paths turned into what Ollama takes.

    Ollama wants base64 under `images` on the message itself. The session
    carries paths instead, because a transcript is a file somebody may open and
    a megabyte of base64 in it helps nobody. The conversion happens here, on the
    way out, once per request.

    An unreadable path is dropped rather than raised: a screenshot deleted
    between being pasted and being sent should cost the image, not the turn.
    """
    paths = message.get("images")
    if not paths:
        return message
    from .clipboard import encode

    encoded = []
    for path in paths:
        try:
            encoded.append(encode(path))
        except OSError:
            continue
    sent = {key: value for key, value in message.items() if key != "images"}
    if encoded:
        sent["images"] = encoded
    return sent


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
        #: Asked of the server the first time it matters; see can_see.
        self._vision: bool | None = None
        self.timeout = httpx.Timeout(
            timeout, connect=CONNECT_TIMEOUT, write=CONNECT_TIMEOUT
        )

    def can_see(self) -> bool:
        """Whether this model accepts images.

        Asked of the server rather than guessed from the name: Ollama lists
        `vision` among a model's capabilities, and a list of model names that
        can see would be out of date the week after it was written.

        Answered once and remembered. Anything that goes wrong -- no server, an
        older Ollama with no `capabilities` -- is False, because the caller's
        next move is to tell the user the picture will not be looked at, and
        that is the true answer in every one of those cases.
        """
        if self._vision is None:
            self._vision = self._ask_capabilities()
        return self._vision

    def _ask_capabilities(self) -> bool:
        try:
            response = httpx.post(
                f"{self.host}/api/show",
                json={"model": self.model},
                timeout=CONNECT_TIMEOUT,
            )
            if response.status_code != 200:
                return False
            return "vision" in (response.json().get("capabilities") or [])
        except (httpx.HTTPError, ValueError):
            return False

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[Chunk]:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_with_images(message) for message in messages],
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

    def deterministic(self) -> "OllamaProvider":
        """A copy of this provider that samples at zero; see :func:`for_review`."""
        return OllamaProvider(
            model=self.model,
            host=self.host,
            num_ctx=self.num_ctx,
            temperature=0.0,
            keep_alive=self.keep_alive,
        )


def _local():
    from .openai import OpenAIProvider

    return OpenAIProvider


def _ollama():
    return OllamaProvider


def _claude_code():
    from .external import ClaudeCodeProvider

    return ClaudeCodeProvider


def _codex():
    from .external import CodexProvider

    return CodexProvider


#: Every backend that can be named in ``config.provider``, by that name.
#:
#: The values are loaders rather than classes so that naming a backend is what
#: imports it. The error for a name that is not here lists what is, rather than
#: failing at the first turn.
BACKENDS = {
    "local": _local,
    "ollama": _ollama,
    "claude-code": _claude_code,
    "codex": _codex,
}

#: The defaults that depend on which backend is running.
#:
#: A model tag, a server URL and a window size all mean something different to a
#: local server than to a hosted API, so `provider` cannot be switched on its
#: own without these moving with it. ``config`` reads this for any of the three
#: the user has not set themselves, which is what makes switching one command
#: rather than four. An empty host means "wherever the backend goes by default".
DEFAULTS = {
    "local": {
        "model": OPENAI_MODEL, "host": OPENAI_HOST, "num_ctx": DEFAULT_NUM_CTX,
    },
    "ollama": {"model": DEFAULT_MODEL, "host": DEFAULT_HOST, "num_ctx": DEFAULT_NUM_CTX},
    "claude-code": {
        "model": DEFAULT_CLAUDE_CODE_MODEL, "host": "", "num_ctx": CLAUDE_CODE_NUM_CTX,
    },
    "codex": {"model": DEFAULT_CODEX_MODEL, "host": "", "num_ctx": CODEX_NUM_CTX},
}


#: Where a backend's fallback goes when nothing is serving it. Only the
#: built-in default has one: a backend the user asked for by name must fail
#: loudly, because silently running a different model than the one somebody
#: typed is worse than not running.
FALLBACK = {"local": "ollama"}


def reachable(name: str, host: str) -> bool:
    """Whether ``name`` has something answering at ``host`` right now.

    Cheap and forgiving. It runs before every default session, so it is one
    request with a short timeout, and anything that goes wrong is False --
    which sends the caller to the fallback, and the fallback is the safe answer
    to every one of those cases.
    """
    try:
        return build(name, host=host).available()
    except (ProviderError, TypeError, ValueError):
        return False


def settle(name: str, host: str) -> tuple[str, str]:
    """The backend to actually run, and a sentence when it is not ``name``.

    The default is `local`, which is right for a machine serving a model it
    trained and wrong for one that has only ever run Ollama -- and a first
    session that fails to connect teaches nothing except that this does not
    work. So when nothing answers on the default endpoint and Ollama does, the
    session runs on Ollama and says so.

    Only reached for a backend nobody named. `--provider local` against a dead
    server is an error, and it stays one: quietly answering with a different
    model than the one somebody asked for is the worse failure.
    """
    alternative = FALLBACK.get(name)
    if alternative is None or reachable(name, host):
        return name, ""
    if not reachable(alternative, DEFAULTS[alternative]["host"]):
        # Neither is up. Staying on the default keeps the error about the thing
        # that was actually configured, and `coder doctor` explains it.
        return name, ""
    return alternative, (
        f"Nothing is serving on {host}, so this session is running on "
        f"{alternative}. Start a server there, or pin one with "
        f"`coder config set provider {alternative}`."
    )


def build(name: str = DEFAULT_PROVIDER, **options) -> Provider:
    """The provider called ``name``, constructed with ``options``."""
    loader = BACKENDS.get(name)
    if loader is None:
        raise ProviderError(
            f"unknown provider {name!r}; available: {', '.join(sorted(BACKENDS))}"
        )
    return loader()(**options)


def for_review(provider: Provider) -> Provider:
    """A copy of ``provider`` that samples deterministically.

    Review is a measurement, not a conversation. At the default temperature the
    same diff yields findings on one run and an empty array on the next, which
    makes recall and precision unusable as a metric -- a prompt change and a
    dice roll look identical.

    Asked of the provider rather than decided here by type. A backend that can
    turn sampling off says so by offering ``deterministic``; one that cannot --
    a fake, or a frontier model behind somebody else's command line -- is
    returned unchanged, which is the only honest thing to do about it.
    """
    pin = getattr(provider, "deterministic", None)
    return pin() if callable(pin) else provider
