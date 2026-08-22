"""The agent loop.

Send history, parse tool calls out of the reply, execute them, repeat until the
model answers in prose or the iteration cap is reached.

Malformed-call handling is core rather than an edge case. Because tool calls
travel in message content, a badly formatted call is the *common* path on a 14b
model, not a rare one. Every failure -- unknown tool, schema violation, tool
error, or a reply with neither a call nor an answer -- is fed back as a `tool`
message the model can correct from. This single behaviour is the difference
between usable and infuriating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import prompts
from .context import compact, should_compact
from .language import detect as detect_language
from .parsing import ToolCall
from .provider import Provider, ProviderError, Reply, collect
from .retrieval import scout, terms
from .session import Session
from .tools.base import Registry, ToolError, ToolResult, validate_arguments

MAX_ITERATIONS = 25
MAX_RETRIES = 2


class Listener(Protocol):
    """Receives loop events so the CLI can render them as they happen."""

    def on_token(self, text: str) -> None: ...
    def on_tool_call(self, call: ToolCall) -> None: ...
    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None: ...
    def on_retry(self, reason: str) -> None: ...


class NullListener:
    """Default listener: renders nothing. Used by tests and by review passes."""

    def on_token(self, text: str) -> None:
        pass

    def on_tool_call(self, call: ToolCall) -> None:
        pass

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        pass

    def on_retry(self, reason: str) -> None:
        pass


@dataclass
class Outcome:
    """What one :meth:`Agent.run` produced."""

    answer: str = ""
    raw: str = ""  # the final reply before tool-call JSON was stripped
    iterations: int = 0
    tool_calls: int = 0
    stopped: str = "answered"  # answered | iteration-cap | retry-cap | denied
    errors: list[str] = field(default_factory=list)


class Agent:
    """Drives one conversation to a final answer."""

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        session: Session,
        listener: Listener | None = None,
        permissions=None,
        max_iterations: int = MAX_ITERATIONS,
        scout_root: Path | str | None = None,
        track_language: bool = True,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.session = session
        self.listener = listener or NullListener()
        self.permissions = permissions
        self.max_iterations = max_iterations
        # Off for the review passes, whose replies are JSON rather than prose:
        # a reminder to answer in Uzbek would only corrupt them.
        self.track_language = track_language
        # None means no scouting. The review passes drive their own conversation
        # from a diff they already have, so retrieval would only be noise there.
        self.scout_root = Path(scout_root) if scout_root else None

    def run(self, user_message: str) -> Outcome:
        """Run the loop until the model answers, or a bound is hit."""
        self.session.add_user(user_message)
        self._note_language(user_message)
        self._scout(user_message)
        return self.resume()

    def _note_language(self, user_message: str) -> None:
        """Update the language the session is being conducted in.

        A message that identifies no language leaves the previous answer
        standing, which is the point: "rahmat" or a bare path says nothing, and
        dropping the language there would let the next reply revert.
        """
        if not self.track_language:
            return
        detected = detect_language(user_message)
        if detected:
            self.session.language = detected

    def _scout(self, user_message: str) -> None:
        """Search the workspace for what the message is about, before asking.

        The result is recorded as a `tool` message the model never called, which
        is the point: it starts the turn already knowing where to look instead
        of spending two or three of them finding out, or -- more often -- not
        bothering. A failure here is not worth losing the turn over, so it
        degrades to no search at all.
        """
        if self.scout_root is None:
            return

        try:
            wanted = terms(user_message)
            block = scout(self.scout_root, user_message) if wanted else ""
        except Exception:
            return
        if not block:
            return

        # Reported through the ordinary tool events so every listener renders it
        # without needing to know it exists.
        call = ToolCall(name="codebase_search", arguments={"terms": ", ".join(wanted)})
        self.listener.on_tool_call(call)
        self.listener.on_tool_result(call, ToolResult.success(block))
        self.session.add_tool_result(call.name, block)

    def resume(self) -> Outcome:
        """Continue from the current history without adding a user message."""
        outcome = Outcome()
        retries = 0

        for _ in range(self.max_iterations):
            outcome.iterations += 1

            try:
                reply = self._ask()
            except ProviderError as exc:
                outcome.errors.append(str(exc))
                outcome.stopped = "provider-error"
                return outcome

            self.session.record_usage(reply.prompt_tokens, reply.completion_tokens)
            self.session.add_assistant(reply.content)

            if not reply.tool_calls:
                # Prose first, but fall back to the raw content: a reply that is
                # only JSON and is not a tool call is a legitimate final answer,
                # which is exactly the shape the review passes return.
                answer = reply.prose or reply.content.strip()
                if answer:
                    outcome.answer = answer
                    outcome.raw = reply.content
                    outcome.stopped = "answered"
                    return outcome

                # Neither a usable call nor a plausible answer: retry, don't stop.
                retries += 1
                if retries > MAX_RETRIES:
                    outcome.stopped = "retry-cap"
                    outcome.errors.append("model produced no usable reply")
                    return outcome
                reason = prompts.no_call_and_no_answer(self.registry.names())
                self.listener.on_retry(reason)
                self.session.add_tool_result("system", reason)
                continue

            progressed = False
            for call in reply.tool_calls:
                outcome.tool_calls += 1
                self.listener.on_tool_call(call)
                result = self._execute(call)
                self.listener.on_tool_result(call, result)
                self.session.add_tool_result(call.name, result.as_message())

                if result.ok:
                    progressed = True
                else:
                    outcome.errors.append(result.error)
                    if result.error.startswith("permission denied"):
                        outcome.stopped = "denied"
                        return outcome

            # A round where every call failed is a retry, and retries are bounded
            # so a model stuck on a bad format cannot spin to the iteration cap.
            if progressed:
                retries = 0
            else:
                retries += 1
                if retries > MAX_RETRIES:
                    outcome.stopped = "retry-cap"
                    return outcome

        outcome.stopped = "iteration-cap"
        return outcome

    def _ask(self) -> Reply:
        """One streamed round trip, with tokens forwarded to the listener."""
        self._compact_if_needed()

        def stream():
            for chunk in self.provider.chat(
                self.session.payload(), self.registry.declarations()
            ):
                if chunk.content:
                    self.listener.on_token(chunk.content)
                yield chunk

        return collect(stream())

    def _compact_if_needed(self) -> None:
        """Summarize the oldest turns once the window is close to full."""
        num_ctx = getattr(self.provider, "num_ctx", 0)
        if not should_compact(self.session, num_ctx):
            return
        if compact(self.session, self.provider) is not None:
            self.listener.on_retry("compacted earlier turns to free context")

    def _execute(self, call: ToolCall) -> ToolResult:
        """Validate, authorize, and run one tool call.

        Never raises: every failure comes back as a :class:`ToolResult` whose
        text tells the model what to do differently.
        """
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult.failure(
                prompts.malformed_call(
                    f"there is no tool named '{call.name}'", self.registry.names()
                )
            )

        try:
            arguments = validate_arguments(tool, call.arguments)
        except ToolError as exc:
            return ToolResult.failure(
                prompts.malformed_call(str(exc), self.registry.names())
            )

        if self.permissions is not None:
            decision = self.permissions.check(tool, arguments)
            if not decision.allowed:
                return ToolResult.failure(f"permission denied: {decision.reason}")

        try:
            return tool.run(**arguments)
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        except Exception as exc:  # a tool bug must not kill the session
            return ToolResult.failure(f"{tool.name} failed unexpectedly: {exc}")
