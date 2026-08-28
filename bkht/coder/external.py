"""Backends that borrow a model from another agent's command line.

Claude Code and Codex are both installed, both already logged in, and both will
answer a question on stdin and stream the reply back as JSON. That makes them
usable as transports -- the same job ``OllamaProvider`` does, done by a program
instead of an HTTP endpoint -- and it means switching to a frontier model costs
no API key, no billing setup, and no new dependency.

**They are used as transports and nothing more.** Each of these tools is a
complete coding agent with its own loop, its own file tools and its own idea of
what it may touch. None of that is wanted here: coder has a permission gate, a
snapshot store and an undo command, and an edit made behind them is an edit the
user cannot take back. So both are launched with their own tooling shut off
(``--tools ""`` for Claude Code, a read-only sandbox for Codex) and asked only
to produce text. Every tool call in that text is still executed by coder,
through the same gate the local model's calls go through.

Two costs come with this, and neither is hidden by anything below. The work
leaves the machine -- the whole point of the Ollama default is that it does not
-- and each turn pays a process launch, because these tools have no
conversation to resume between them. Coder resends its history every turn
anyway, which is exactly what makes a stateless launch correct rather than
merely tolerable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Iterator

from .provider import (
    CLAUDE_CODE_NUM_CTX,
    CODEX_NUM_CTX,
    DEFAULT_CLAUDE_CODE_MODEL,
    DEFAULT_CODEX_MODEL,
    Chunk,
    ProviderError,
)

#: A turn that has produced nothing for this long is not thinking, it is stuck.
#: Generous rather than tight: a frontier model on a hard turn with reasoning on
#: legitimately spends minutes before its first token.
TIMEOUT = 600.0

#: Labels a tool result carries into the prompt. The same shape
#: :func:`context.transcript` uses, so the model sees its own history rendered
#: the way the summarizer would render it.
ROLES = {"user": "user", "assistant": "assistant"}


def render(messages: list[dict]) -> tuple[str, str]:
    """Coder's message list as ``(system, prompt)``.

    These tools take one prompt, not a conversation, so the history is flattened
    into it. That is not a downgrade: coder is stateless across turns by
    design -- it resends everything every time -- so a flattened history and a
    replayed one contain exactly the same thing.
    """
    system: list[str] = []
    lines: list[str] = []

    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        # An image is named rather than sent. Both of these tools are perfectly
        # able to open a file, and both are launched with their tooling shut
        # off -- but a path in the prompt is a thing the user can ask them to
        # look at, where a base64 blob in a text field is not.
        if paths := message.get("images"):
            named = "\n".join(f"[image] {path}" for path in paths)
            content = f"{content}\n{named}".strip() if content else named
        if not content:
            continue
        if role == "system":
            system.append(content)
        elif role == "tool":
            lines.append(f"[{message.get('name') or 'tool'}] {content}")
        else:
            lines.append(f"[{ROLES.get(role, 'user')}] {content}")

    return "\n\n".join(system), "\n\n".join(lines)


class CommandProvider:
    """A provider that runs a command and reads JSON lines back from it.

    Subclasses supply the two halves that differ: how to build the argument
    list, and what one decoded line means. Everything else -- launching,
    feeding the prompt in, streaming the reply out, and turning a non-zero exit
    into something a user can act on -- is the same for both tools.
    """

    #: The executable, and the name to use when it is not on PATH.
    command = ""

    def can_see(self) -> bool:
        """True: both of these tools can open an image file for themselves.

        Not the same "yes" the Ollama backend gives. Nothing is attached to the
        request -- the path is named in the prompt, and the tool at the other
        end reads it -- but from the user's side the picture does get looked at,
        which is the question being asked.
        """
        return True
    install_hint = ""

    def __init__(
        self,
        model: str = "",
        host: str = "",
        num_ctx: int = 0,
        temperature: float | None = None,
        timeout: float = TIMEOUT,
    ) -> None:
        self.model = model
        # Carried so that every setting has somewhere to land, and unused: there
        # is no server to point at and no sampling knob to turn on a command
        # line that does not offer one.
        self.host = host
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def argv(self, system: str) -> list[str]:
        """The arguments after the executable itself."""
        raise NotImplementedError

    def decode(self, event: dict) -> Chunk | None:
        """One JSON line as a chunk, or None for a line that says nothing."""
        raise NotImplementedError

    def executable(self) -> str:
        """The tool's path, or a ``ProviderError`` saying how to get it."""
        found = shutil.which(self.command)
        if found is None:
            raise ProviderError(f"{self.command} is not on PATH. {self.install_hint}")
        return found

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[Chunk]:
        """One turn, streamed.

        ``tools`` is accepted and ignored, exactly as it is for Ollama: coder
        states its own tool protocol in the system prompt and parses the calls
        back out of the content, so a second, native protocol would only give
        the model two contradicting sets of instructions.
        """
        system, prompt = render(messages)
        if not prompt:
            raise ProviderError("there is nothing to send: every message was empty")

        argv = [self.executable(), *self.argv(system)]
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # A pager or a colour code in the middle of a JSON stream would be
            # a parse error with no useful message attached to it.
            env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
        )

        try:
            yield from self._pump(process, prompt)
        except (ProviderError, GeneratorExit):
            # A turn abandoned halfway -- a refusal, or a user interrupt -- must
            # not leave a frontier model running in the background billing.
            process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def _pump(self, process, prompt: str) -> Iterator[Chunk]:
        """Feed the prompt in, read chunks out, and check how it ended."""
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except OSError as exc:
            raise ProviderError(f"{self.command} would not take the prompt: {exc}") from exc

        for line in process.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = self.decode(event)
            if chunk is not None:
                yield chunk

        if process.wait() != 0:
            detail = (process.stderr.read() or "").strip()
            raise ProviderError(
                f"{self.command} exited {process.returncode}"
                + (f": {detail[:400]}" if detail else "")
            )

    def available(self) -> bool:
        """Whether the tool is installed, used to skip live tests."""
        return shutil.which(self.command) is not None


class ClaudeCodeProvider(CommandProvider):
    """Claude Code in print mode, with its own tools switched off."""

    command = "claude"
    install_hint = "Install it from https://claude.com/claude-code, then run `claude` once to log in."

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_CODE_MODEL,
        num_ctx: int = CLAUDE_CODE_NUM_CTX,
        **options,
    ) -> None:
        super().__init__(model=model, num_ctx=num_ctx, **options)

    def argv(self, system: str) -> list[str]:
        """The flags that turn an agent into a transport.

        ``--tools ""`` is the load-bearing one: it leaves Claude Code with
        nothing to act with, so the only thing it can return is the text coder
        asked for. The rest is quieting -- ``--restricted`` skips the user's own
        settings and hooks, and the two MCP and skill switches keep a personal
        setup from being loaded into somebody else's agent.
        """
        return [
            "--print",
            "--restricted",
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--model", self.model,
            "--system-prompt", system,
        ]

    def decode(self, event: dict) -> Chunk | None:
        kind = event.get("type")

        if kind == "stream_event":
            inner = event.get("event") or {}
            delta = inner.get("delta") or {}
            if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                return Chunk(content=delta.get("text") or "")
            return None

        if kind == "result":
            if event.get("is_error"):
                raise ProviderError(f"claude: {str(event.get('result') or '')[:400]}")
            return Chunk(done=True, **_counts(event.get("usage") or {}))

        return None


class CodexProvider(CommandProvider):
    """Codex in exec mode, sandboxed read-only.

    Codex has no switch that removes its shell tool, so the guard here is the
    sandbox rather than an empty tool list: whatever it decides to run, it
    cannot write. Every change still arrives as text for coder to execute.
    """

    command = "codex"
    install_hint = "Install it with `npm i -g @openai/codex`, then run `codex` once to log in."

    def argv(self, system: str) -> list[str]:
        """Codex has no system-prompt flag, so the prompt carries it.

        Everything else is isolation: no session file left behind, no personal
        config, no execpolicy rules, and no requirement that the workspace be a
        git repository -- coder is already deciding all of that for itself.
        """
        return [
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--model", self.model,
            "-",
        ]

    def __init__(
        self,
        model: str = DEFAULT_CODEX_MODEL,
        num_ctx: int = CODEX_NUM_CTX,
        **options,
    ) -> None:
        super().__init__(model=model, num_ctx=num_ctx, **options)
        #: The text already emitted per item, so a re-reported item streams the
        #: part that is new rather than repeating itself.
        self._seen: dict[str, str] = {}

    def chat(self, messages, tools=None):
        # Codex has no system-prompt flag, so the instructions become the first
        # user turn. Prepended rather than appended: they are the instructions,
        # and instructions are read before the work they describe.
        promoted = [
            {"role": "user", "content": message.get("content")}
            if message.get("role") == "system" else message
            for message in messages
        ]
        self._seen.clear()
        yield from super().chat(promoted, tools)

    def decode(self, event: dict) -> Chunk | None:
        kind = event.get("type")

        if kind in ("error", "turn.failed"):
            message = event.get("message") or (event.get("error") or {}).get("message")
            raise ProviderError(f"codex: {str(message or 'the turn failed')[:400]}")

        # Codex reports whole items rather than token deltas, and reports the
        # same item again as it grows. Only the part that is new is emitted, so
        # a tool that streams and one that does not both come out once.
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item") or {}
            if item.get("type") != "agent_message":
                return None
            item_id = str(item.get("id"))
            text = item.get("text") or ""
            seen = self._seen.get(item_id, "")
            self._seen[item_id] = text
            fresh = text[len(seen):] if text.startswith(seen) else text
            return Chunk(content=fresh) if fresh else None

        if kind == "turn.completed":
            return Chunk(done=True, **_counts(event.get("usage") or {}))

        return None



def _counts(usage: dict) -> dict:
    """Prompt and completion tokens, however the tool spells them.

    Cached input is counted as input. It is cheaper, not absent, and the number
    coder wants is how full the window is -- a history that is mostly cache
    hits takes up exactly as much room as one that is not.
    """
    prompt = sum(
        int(usage.get(key) or 0)
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cached_input_tokens",
        )
    )
    completion = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt or None,
        "completion_tokens": completion or None,
    }
