"""User commands fired on tool events.

`permissions.json` remembers what was allowed; it cannot *do* anything. A hook
can: run the formatter after a write, refuse a shell command whose shape this
project never wants, kick off a build when the turn ends.

Three events, and only one of them can say no:

``pre_tool``
    Before a call runs, after it has passed validation and permission. A
    non-zero exit **blocks** the call, and what the hook printed becomes the
    tool result -- so the model reads the refusal and adapts, which is the same
    correction path a malformed call or a denied permission already takes.
``post_tool``
    After a call has run, whatever it returned. Its exit code is reported and
    ignored: the call already happened, and there is nothing left to refuse.
``turn_end``
    Once, after the turn has stopped, however it stopped.

Hooks are read from ``config.json`` beside the settings, as ``event -> list of
commands``. They are shell commands, run in the workspace root, so **a hook is
arbitrary code from a config file** -- see ``SECURITY.md``, and see ``coder
doctor``, which lists every hook it can find precisely so that none of them is
invisible.

Everything a hook might want is in the environment rather than the command
line, because a command line means quoting and quoting means a hook that
silently does the wrong thing on a path with a space in it.

Every hook is timed out. A formatter that hangs must not be indistinguishable
from a model that hangs -- and, for ``pre_tool`` only, a timeout blocks the
call it was asked to rule on. That is the one place failing open would be
worse than failing loudly: a gate nobody heard from is not a gate.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"
TURN_END = "turn_end"
EVENTS = (PRE_TOOL, POST_TOOL, TURN_END)

#: Long enough for a formatter or a linter, short enough that a hung one is
#: over before the user reaches for Esc -- which does not reach this thread
#: anyway, for the same `waitpid` reason `verify.suite` documents.
TIMEOUT = 30.0
#: What a hook says, as fed back to the model. A blocked call's reason has to
#: fit in a tool result beside everything else the turn is carrying.
OUTPUT_LIMIT = 2_000
#: Arguments are passed as JSON in the environment, and an environment is not
#: the place to put a whole file's contents.
ARGS_LIMIT = 4_000


@dataclass(frozen=True)
class Result:
    """What one hook did."""

    event: str
    command: str
    code: int | None = None
    output: str = ""
    #: Set when nothing could be spawned at all -- there is no shell, or the
    #: process could not be started. Narrower than it sounds: a command the
    #: shell cannot find is not this, it is an ordinary exit 127, and a
    #: `pre_tool` gate whose script has been deleted therefore blocks. That is
    #: the right way round for a gate, and it is why this field is only about
    #: the shell itself.
    broken: str = ""
    timed_out: bool = False
    #: What the timeout actually was, carried so a report of one names the
    #: number that was enforced rather than the module's default. They differ
    #: whenever a caller passes its own.
    after: float = TIMEOUT

    @property
    def blocked(self) -> bool:
        """Whether this result stops the call it was fired for.

        Only ``pre_tool``, and only where there was something to hear from: a
        machine with no shell at all has not refused anything, and blocking
        every call on it would turn a config file nobody can run into an agent
        that cannot work. A command the shell looked for and did not find is a
        different case -- that is an exit code, and a gate whose script has
        gone missing is a gate that should fail closed.
        """
        if self.event != PRE_TOOL:
            return False
        return self.timed_out or (self.code is not None and self.code != 0)

    @property
    def reason(self) -> str:
        """Why the call was blocked, in the hook's own words where it has any."""
        if self.timed_out:
            return f"`{self.command}` did not finish within {self.after:.0f}s"
        return self.output.strip() or f"`{self.command}` exited {self.code}"

    def summary(self) -> str:
        """One line, for the user watching the turn."""
        if self.broken:
            return f"hook `{self.command}` could not run: {self.broken}"
        if self.timed_out:
            return f"hook `{self.command}` timed out after {self.after:.0f}s"
        if self.code:
            return f"hook `{self.command}` exited {self.code}"
        return f"hook `{self.command}` ok"


def _text(*parts: str) -> str:
    """The halves of a hook's output, joined, longest-lived part first."""
    joined = "\n".join(part.rstrip() for part in parts if part and part.strip())
    if len(joined) > OUTPUT_LIMIT:
        joined = joined[:OUTPUT_LIMIT] + "\n[truncated]"
    return joined


def parse(raw: object) -> tuple[dict[str, list[str]], list[str]]:
    """``event -> commands`` out of whatever the config file actually held.

    Returns the problems alongside rather than raising, because this is read on
    the way into a session: a malformed hooks block should cost the user their
    hooks and a printed sentence, not the session that would let them fix it.
    """
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        return {}, ["hooks must be an object of event -> commands"]

    parsed: dict[str, list[str]] = {}
    problems: list[str] = []
    for event, value in raw.items():
        if event not in EVENTS:
            problems.append(f"unknown hook event {event!r}; known: {', '.join(EVENTS)}")
            continue
        commands = [value] if isinstance(value, str) else value
        if not isinstance(commands, list):
            problems.append(f"hooks.{event} must be a command or a list of commands")
            continue
        kept = [c.strip() for c in commands if isinstance(c, str) and c.strip()]
        if len(kept) != len(commands):
            problems.append(f"hooks.{event}: ignored an entry that was not a command")
        if kept:
            parsed[event] = kept
    return parsed, problems


@dataclass
class Hooks:
    """The configured commands, and the machinery to fire them.

    A ``Hooks`` with nothing configured is the common case and costs nothing:
    :meth:`fire` returns before it looks at a shell.
    """

    commands: dict[str, list[str]] = field(default_factory=dict)
    root: Path | str = "."
    timeout: float = TIMEOUT
    #: Injected so the suite can assert what would be run without running it.
    runner: object = subprocess.run

    def __bool__(self) -> bool:
        return any(self.commands.get(event) for event in EVENTS)

    def for_event(self, event: str) -> list[str]:
        return list(self.commands.get(event, ()))

    def listing(self) -> list[tuple[str, str]]:
        """``(event, command)`` in event order, for ``doctor`` to print."""
        return [(e, c) for e in EVENTS for c in self.for_event(e)]

    def fire(self, event: str, **context) -> list[Result]:
        """Run every command for ``event``, in the order they were written."""
        commands = self.for_event(event)
        if not commands:
            return []

        from .tools.shell import resolve_shell

        shell = resolve_shell()
        environment = _environment(event, self.root, context)
        results = []
        for command in commands:
            if shell is None:
                results.append(Result(event, command, broken="no shell is available"))
                continue
            results.append(self._one(event, command, shell, environment))
        return results

    def _one(self, event: str, command: str, shell, environment: dict) -> Result:
        try:
            completed = self.runner(
                [*shell.argv, command],
                cwd=str(self.root),
                env=environment,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(event, command, timed_out=True, after=self.timeout)
        except OSError as exc:
            return Result(event, command, broken=str(exc))

        # stderr first here, the opposite of a test run: a hook that refuses a
        # call says why on stderr, and that sentence is the whole result.
        return Result(
            event,
            command,
            code=completed.returncode,
            output=_text(completed.stderr or "", completed.stdout or ""),
        )


def _environment(event: str, root, context: dict) -> dict:
    """The parent environment plus what this event knows.

    Inherited rather than replaced: a hook is the user's own command, and a
    formatter that cannot see ``PATH`` is a formatter that does not run.
    """
    environment = dict(os.environ)
    environment["CODER_EVENT"] = event
    environment["CODER_ROOT"] = str(root)

    tool = context.get("tool")
    if tool:
        environment["CODER_TOOL"] = str(tool)

    arguments = context.get("arguments")
    if isinstance(arguments, dict):
        environment["CODER_ARGS"] = json.dumps(arguments, sort_keys=True)[:ARGS_LIMIT]
        # Lifted out of the JSON because every hook anybody actually writes --
        # format this, lint this -- wants exactly this one value, and making
        # each of them parse JSON in a shell would make hooks a thing only
        # people who like `jq` can use.
        path = arguments.get("path")
        if isinstance(path, str) and path:
            environment["CODER_PATH"] = path

    for key in ("ok", "stopped", "edited", "tool_calls"):
        if key in context and context[key] is not None:
            value = context[key]
            if isinstance(value, bool):
                value = "1" if value else "0"
            environment[f"CODER_{key.upper()}"] = str(value)
    return environment
