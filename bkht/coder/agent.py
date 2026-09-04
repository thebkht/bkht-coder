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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import hooks, prompts, verify
from .cancel import interruptible
from .context import compact, elide_tool_results, should_compact
from .language import detect as detect_language
from .parsing import ToolCall
from .provider import Provider, ProviderError, Reply, collect
from .retrieval import scout, terms
from .session import Session
from .tools.base import Registry, ToolError, ToolResult, truncate, validate_arguments

MAX_ITERATIONS = 25
MAX_RETRIES = 2

#: How many times a turn may ask for a call it has already made before the turn
#: is declared stuck. Two is a model that lost something and reached for it
#: again, which is ordinary; a third is a model going round.
MAX_REPLAYS = 2

#: How much of a replayed result to hand back. Generous, because the point is
#: to remove the reason to invent one, and a truncated answer is a reason.
REPLAY_LIMIT = 4000

#: How long one turn may run before it is asked to answer with what it has.
#: Nothing else bounds the clock: the iteration cap counts round trips, and the
#: retry cap only counts rounds where every call failed, so a turn making
#: distinct successful calls that go nowhere is bounded by neither. One in the
#: session that prompted this ran nineteen minutes and answered nothing.
#:
#: Checked between iterations, never mid-call, so it can overshoot by one round
#: trip. Bounding the model's thinking is not this budget's job -- Esc is.
MAX_SECONDS = 600.0


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
    #: answered | iteration-cap | retry-cap | looping | denied | provider-error
    #: | time-cap | interrupted
    stopped: str = "answered"
    errors: list[str] = field(default_factory=list)
    #: What this turn cost, for the line printed under the answer. Per-turn,
    #: which is what makes it different from the running totals the session
    #: keeps for the context meter: those only ever grow.
    seconds: float = 0.0
    sent: int = 0  # the prompt at its largest, on the last round trip
    received: int = 0  # every token the model generated, across every round


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
        max_seconds: float = MAX_SECONDS,
        scout_root: Path | str | None = None,
        track_language: bool = True,
        verify_command: str = "",
        verify_root: Path | str | None = None,
        hooks=None,
        clock=time.monotonic,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.session = session
        self.listener = listener or NullListener()
        self.permissions = permissions
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        # Injected so a turn's duration can be asserted without sleeping.
        self.clock = clock
        # Off for the review passes, whose replies are JSON rather than prose:
        # a reminder to answer in Uzbek would only corrupt them.
        self.track_language = track_language
        # None means no scouting. The review passes drive their own conversation
        # from a diff they already have, so retrieval would only be noise there.
        self.scout_root = Path(scout_root) if scout_root else None
        # "" means nothing is run. The command is the user's, written down in
        # their config; there is no inferred default here, because the case for
        # running anything at all rests on the command being one they chose.
        self.verify_command = verify_command.strip()
        self.verify_root = Path(verify_root) if verify_root else self.scout_root
        # None rather than an empty Hooks so the common case -- nobody has
        # configured any -- is one identity check per call rather than a dict
        # lookup and a shell resolution.
        self.hooks = hooks or None
        # All reset at the top of every turn; see _loop.
        self._summarized = False
        self._made: dict[tuple[str, str], str] = {}
        self._replays = 0
        self._wrote = False
        self._edited = False
        self._verified = 0

    def run(self, user_message: str, images: list[str] | None = None) -> Outcome:
        """Run the loop until the model answers, or a bound is hit."""
        started = self.clock()
        self.session.add_user(user_message, images=images)
        self._note_language(user_message)
        self._scout(user_message)
        # `_loop` directly rather than through `resume`, so the turn is recorded
        # once, with the duration that includes the scouting above.
        return self._run_to_finish(started)

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
        started = self.clock()
        return self._run_to_finish(started)

    def _run_to_finish(self, started: float) -> Outcome:
        """:meth:`_loop`, recorded however it ends -- Esc included.

        The REPL catches the ``KeyboardInterrupt`` and abandons the task, which
        is right, but it catches it outside :meth:`_finish` -- so an
        interrupted turn was never written down at all. A captured session
        shows what that leaves: a user message, the scout's block, and then the
        next user message, with no outcome between them and nothing saying a
        question had been asked. Unreadable afterwards, and worse to anything
        reading sessions back -- `ingest` segments turns on outcomes, so the
        abandoned request became a hole in the middle of a trajectory, teaching
        the model that a question followed by a search result may be ignored.

        Re-raised once recorded, so the REPL still prints `[interrupted]` and
        nothing about the interrupt itself changes.
        """
        try:
            outcome = self._loop()
        except KeyboardInterrupt:
            outcome = Outcome(stopped="interrupted")
            outcome.errors.append("interrupted by the user")
            # Best effort: the turn is already being abandoned, and failing to
            # write it down is not a reason to lose the Esc as well.
            try:
                self._finish(outcome, started)
            except Exception:
                pass
            raise
        return self._finish(outcome, started)

    def _finish(self, outcome: Outcome, started: float) -> Outcome:
        """Stamp a turn's duration and write down how it ended.

        The transcript already says what was said. This is the part that says
        whether it worked -- which is what anything choosing trajectories to
        learn from needs, and what the message list alone cannot tell it: a turn
        that answered and a turn that hit the iteration cap look the same there.
        """
        outcome.seconds = max(0.0, self.clock() - started)
        self.session.record_outcome(outcome)
        self._fire(
            hooks.TURN_END,
            stopped=outcome.stopped,
            edited=self._edited,
            tool_calls=outcome.tool_calls,
        )
        return outcome

    def _loop(self) -> Outcome:
        """The loop itself. Every exit from here is a finished turn."""
        outcome = Outcome()
        retries = 0
        # Summarizing is allowed once per turn; see _compact_if_needed.
        self._summarized = False
        # Every call already made this turn, with what it returned, so an exact
        # repeat can be answered from here rather than run again.
        self._made: dict[tuple[str, str], str] = {}
        self._replays = 0
        # Whether anything has been written since the last check, and how many
        # checks this turn has spent. Both per-turn: a turn that only read has
        # nothing to check, and last turn's edits were checked last turn.
        self._wrote = False
        # `_wrote` is cleared each time the suite runs -- it means "since the
        # last check". A hook that wants to know whether the turn changed
        # anything at all needs the question that is never un-asked.
        self._edited = False
        self._verified = 0

        deadline = self.clock() + self.max_seconds
        for _ in range(self.max_iterations):
            if outcome.iterations and self.clock() > deadline:
                outcome.stopped = "time-cap"
                outcome.errors.append(
                    f"turn ran past {int(self.max_seconds)}s without answering"
                )
                self._final_answer(outcome)
                return outcome
            outcome.iterations += 1

            try:
                reply = self._ask()
            except ProviderError as exc:
                outcome.errors.append(str(exc))
                outcome.stopped = "provider-error"
                return outcome

            self.session.record_usage(reply.prompt_tokens, reply.completion_tokens)
            outcome.sent = reply.prompt_tokens or outcome.sent
            outcome.received += reply.completion_tokens or 0
            # An empty reply is recorded nowhere. It carries nothing the model
            # could act on, and leaving it in history teaches the format that
            # produced it: three attempts at a malformed call used to mean the
            # model read two of its own failures before making the third.
            if reply.content.strip():
                self.session.add_assistant(reply.content)

            if not reply.tool_calls:
                # Prose first, but fall back to the raw content: a reply that is
                # only JSON and is not a tool call is a legitimate final answer,
                # which is exactly the shape the review passes return.
                answer = reply.prose or reply.content.strip()
                if answer:
                    # The one moment the turn is known to have finished
                    # writing. Running the suite after every write would spend
                    # the iteration budget on the test runner; running it here
                    # costs one run for a turn that edited, and none at all for
                    # a turn that only read.
                    if self._check_work(outcome):
                        continue
                    outcome.answer = answer
                    outcome.raw = reply.content
                    outcome.stopped = "answered"
                    return outcome

                # Neither a usable call nor a plausible answer: retry, don't stop.
                retries += 1
                if retries > MAX_RETRIES:
                    outcome.stopped = "retry-cap"
                    outcome.errors.append("model produced no usable reply")
                    self._final_answer(outcome)
                    return outcome
                self.listener.on_retry("retrying: no tool call and no answer")
                self.session.add_tool_result(
                    "system", prompts.no_call_and_no_answer(self.registry.names())
                )
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
                    tool = self.registry.get(call.name)
                    if tool is not None and tool.mutating:
                        self._wrote = True
                        self._edited = True
                else:
                    outcome.errors.append(result.error)
                    if result.error.startswith("permission denied"):
                        outcome.stopped = "denied"
                        return outcome

            # A model that keeps asking for calls it has already made is going
            # round, not working. The replay gives it back what it asked for, so
            # a second one is still ordinary -- a third means the answers are not
            # what it is short of, and no further iteration will change that.
            if self._replays > MAX_REPLAYS:
                outcome.stopped = "looping"
                outcome.errors.append("model kept asking for calls it had already made")
                self._final_answer(outcome)
                return outcome

            # A round where every call failed is a retry, and retries are bounded
            # so a model stuck on a bad format cannot spin to the iteration cap.
            if progressed:
                retries = 0
            else:
                retries += 1
                if retries > MAX_RETRIES:
                    outcome.stopped = "retry-cap"
                    self._final_answer(outcome)
                    return outcome

        outcome.stopped = "iteration-cap"
        self._final_answer(outcome)
        return outcome

    def _check_work(self, outcome: Outcome, feedback: bool = True) -> bool:
        """Run the project's tests over what this turn wrote. Keep going?

        Returns True when the loop should carry on -- the suite failed and the
        model has been handed the failure to correct from. False means the turn
        is free to end, which covers every case where nothing was run: no
        command configured, nothing written, or the tests passed.

        ``feedback=False`` runs the check and reports it without handing
        anything back, and always returns False. That is the bounded-turn case:
        a turn that hit the iteration cap has no room left to act on a failure,
        but it is also the turn most likely to have stopped halfway through an
        edit -- so whether it left the tests broken is exactly what the person
        reading it needs to know.

        The suite runs *after* the model has said it is finished, which is what
        makes it cheap. A check after every write would put the test runner
        inside the edit loop and spend the iteration budget on it; a turn that
        only read runs nothing at all.

        Bounded at two runs. The first is the check, the second is the fix
        being checked, and a third would mean handing back a failure the model
        has already failed to fix once -- the least promising thing left to
        try, at the price of the iterations a clear account of the problem
        would have cost.
        """
        if not (self.verify_command and self._wrote and self.verify_root):
            return False
        if self._verified >= verify.MAX_RUNS:
            return False

        # Cleared here, not at the end of the turn, so `_wrote` means "wrote
        # since the last check" rather than "wrote at some point". A model that
        # reads a failure and answers without editing has changed nothing, and
        # running the suite again to watch it fail identically spends the
        # timeout budget to learn what the previous run already said.
        self._wrote = False
        self._verified += 1
        self.listener.on_retry(f"running {self.verify_command}")
        report = verify.suite(self.verify_command, self.verify_root)
        self.listener.on_retry(report.summary())

        if report.ok:
            return False
        if not feedback:
            outcome.errors.append(report.summary())
            return False
        if report.status in (verify.TIMED_OUT, verify.BROKEN):
            # Not the model's problem and not something it can fix. The turn
            # ends; the failure is recorded so the CLI can say the check did
            # not happen rather than let a passing silence imply that it did.
            outcome.errors.append(report.summary())
            return False

        output = truncate(report.output, max_chars=verify.OUTPUT_LIMIT)
        last = self._verified >= verify.MAX_RUNS
        say = prompts.suite_still_failing if last else prompts.suite_failed
        self.session.add_tool_result("system", say(self.verify_command, output))
        outcome.errors.append(report.summary())
        return True

    def _final_answer(self, outcome: Outcome) -> None:
        """One last round asking for prose, when the loop ran out of room.

        A bounded turn used to end with nothing at all, and the CLI filled the
        gap with the last tool error -- a message written for the model, shown
        to the user as though it explained why the turn stopped. By the time a
        turn is bounded the model has usually read enough to say something
        useful. It was simply never asked.
        """
        # Reported, never fed back. There is no room left to act on a failure
        # here -- that is what being bounded means -- but a turn that ran out
        # mid-edit is the one most likely to have left the tests broken, and
        # ending in silence about that is worse than the extra run costs.
        self._check_work(outcome, feedback=False)
        self.session.add_tool_result("system", prompts.out_of_steps())
        try:
            reply = self._ask()
        except ProviderError as exc:
            outcome.errors.append(str(exc))
            return

        self.session.record_usage(reply.prompt_tokens, reply.completion_tokens)
        outcome.received += reply.completion_tokens or 0

        answer = reply.prose.strip()
        if not answer:
            return
        self.session.add_assistant(reply.content)
        outcome.answer = answer
        outcome.raw = reply.content

    def _ask(self) -> Reply:
        """One streamed round trip, with tokens forwarded to the listener.

        The tool declarations are deliberately *not* sent. Ollama's qwen2.5
        template renders its own ``<tool_call></tool_call>`` protocol whenever
        ``tools`` is present, which contradicts the one this system prompt
        states -- and measurement says the model follows neither reliably when
        it is given both. Asked with a bare system prompt and ``tools`` set,
        qwen2.5-coder:14b answered with the plain JSON object three times out of
        three and left ``message.tool_calls`` null every time. The native
        transport is a promise the model does not keep, so it is not asked for.
        Parsing for it stays in :func:`provider.collect`, for a model that does.
        """
        self._compact_if_needed()

        # The read is pumped through a worker so that Esc lands during the
        # wait before the first token, which on a local model is most of a
        # turn. Rendering stays on this thread: the listener draws, and two
        # threads drawing is a different bug.
        def rendered():
            for chunk in interruptible(self.provider.chat(self.session.payload())):
                if chunk.content:
                    self.listener.on_token(chunk.content)
                yield chunk

        return collect(rendered())

    def _compact_if_needed(self) -> None:
        """Free context when the window is close to full.

        Summarizing costs a full model call, so a turn gets one. After that the
        pressure is relieved by eliding old tool output instead, which is free.
        Without that limit a single large read put the turn permanently over
        threshold and it summarized on every iteration -- three or four extra
        model calls per turn, each one throwing away the model's record of what
        it had already read, so it read it again. That was the loop the user saw.
        """
        num_ctx = getattr(self.provider, "num_ctx", 0)
        if not should_compact(self.session, num_ctx):
            return

        if not self._summarized:
            self._summarized = True
            if compact(self.session, self.provider) is not None:
                self.listener.on_retry("compacted earlier turns to free context")
                return

        elided = elide_tool_results(self.session)
        if elided:
            self.listener.on_retry(
                f"dropped {elided} earlier tool result"
                f"{'' if elided == 1 else 's'} to free context"
            )

    def _execute(self, call: ToolCall) -> ToolResult:
        """Validate, authorize, and run one tool call.

        Never raises: every failure comes back as a :class:`ToolResult` whose
        text tells the model what to do differently.
        """
        # An exact repeat is refused before anything else runs. Freeing context
        # necessarily costs the model some of what it read, and a model that has
        # lost a file reaches for it again -- which spends the window that made
        # it forget, and loses the file again. Left alone the turn reads the same
        # file until the iteration cap. Refusing is what breaks the cycle: it
        # costs nothing, and the round counts as no progress, so the retry bound
        # ends the turn instead of the iteration cap.
        signature = (call.name, json.dumps(call.arguments, sort_keys=True))
        if signature in self._made:
            self._replays += 1
            return ToolResult.failure(
                prompts.repeated_call(call.name, self._made[signature])
            )

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

        # After permission, not before: a hook is the user's own command, and
        # firing one for a call the user is about to refuse would run it for a
        # call that never happens.
        refused = self._fire(hooks.PRE_TOOL, tool=tool.name, arguments=arguments)
        if refused is not None:
            return ToolResult.failure(f"blocked by a hook: {refused.reason}")

        try:
            result = tool.run(**arguments)
        except ToolError as exc:
            result = ToolResult.failure(str(exc))
        except Exception as exc:  # a tool bug must not kill the session
            result = ToolResult.failure(f"{tool.name} failed unexpectedly: {exc}")

        # Fired for a failed call too. "The write did not happen" is exactly
        # what a hook watching writes needs to hear, and hearing nothing is
        # indistinguishable from not being configured.
        self._fire(hooks.POST_TOOL, tool=tool.name, arguments=arguments, ok=result.ok)
        if not result.ok:
            return result

        # Remembered only once it has actually run, and only its own text: a
        # call that was refused permission or failed to validate never happened,
        # and should not be replayed as though it had.
        self._made[signature] = truncate(result.as_message(), max_chars=REPLAY_LIMIT)
        return result

    def _fire(self, event: str, **context):
        """Run the hooks for ``event``; return the first that blocked, if any.

        A hook that went wrong is reported to the listener the same way a verify
        run is, because a hook that silently rewrote the file the model just
        wrote is a hook that makes the next tool result inexplicable.

        With one exception: a hook that *blocked* is not announced here. Its
        sentence is about to be the tool result, and saying it twice would say
        it twice in the wrong order -- the listener prints a call and its
        result together, so a notice raised mid-call lands above the call it
        was fired for and reads as though it belonged to the previous one.
        """
        if self.hooks is None:
            return None
        blocked = None
        for result in self.hooks.fire(event, **context):
            if result.blocked:
                blocked = blocked or result
                continue
            if result.broken or result.timed_out or result.code:
                self.listener.on_retry(result.summary())
        return blocked
