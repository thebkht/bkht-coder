"""The shared half of the `github` and `gitlab` tools.

Both are the same tool with a different vocabulary: a read-only wrapper around
a CLI that is already installed and already authenticated, so the model can
look at a run, a pull request or an issue without being handed a credential.

Read-only is enforced here rather than left to the permission gate, and that is
the point of the wrapper. `gh` can merge a pull request and delete a release;
those are not things to approve one keypress at a time in the middle of a turn
that was only supposed to read a log. What this cannot do, it cannot be talked
into doing -- and the shell tool is still there, behind the gate, for the times
somebody genuinely means to.

The command is split with :func:`shlex.split` and run as an argument list with
no shell, so there is nothing for a `;` or a backtick to do.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass

from .base import Registry, Tool, ToolError, ToolResult, truncate

#: Long enough for a cold `gh run view --log`, which downloads a zip of the
#: whole run before it prints anything.
TIMEOUT = 120

#: Verbs that change something on the far side. Refused wherever they appear,
#: not merely in second position: `gh pr review --approve` writes, and so does
#: `gh api --method POST`.
WRITES = frozenset({
    "create", "delete", "edit", "merge", "close", "reopen", "rerun", "cancel",
    "upload", "set", "add", "remove", "rename", "transfer", "archive", "sync",
    "lock", "unlock", "pin", "unpin", "approve", "review", "comment", "clone",
    "checkout", "push", "fork", "release", "deploy", "run-workflow", "restore",
    "revoke", "update", "import", "publish", "unarchive", "disable", "enable",
})

#: Flags that turn a read into a write on a CLI that otherwise looks read-only.
WRITE_FLAGS = frozenset({"--method", "-X", "--field", "-F", "--raw-field", "-f"})

#: HTTP methods an `api` call may use.
READ_METHODS = frozenset({"GET", "HEAD"})


@dataclass(frozen=True)
class Forge:
    """One hosting CLI: what to run, and what it is allowed to be asked."""

    name: str
    """The tool name the model sees, and the executable."""

    label: str
    """The service, for the description and for error messages."""

    commands: frozenset[str]
    """Top-level subcommands this tool will pass through."""

    examples: str
    """Lines shown to the model, which is most of how it learns the shape."""

    login: str
    """What to run when the CLI is installed but not authenticated."""


def check(forge: Forge, command: str) -> list[str]:
    """``command`` as an argument list, or raise saying why not.

    Both lists are checked. A deny-list alone would pass every verb the CLI
    gains after this was written; an allow-list alone would pass
    `pr create`, because `pr` is a perfectly good thing to read.
    """
    if not command.strip():
        raise ToolError("command must not be empty")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"could not parse the command: {exc}") from None
    if not argv:
        raise ToolError("command must not be empty")

    # A model that has been told the tool is `github` sometimes writes the
    # executable in as well. Harmless, and worth accepting rather than failing.
    if argv[0] == forge.name:
        argv = argv[1:]
    if not argv:
        raise ToolError("command must not be empty")

    if argv[0] not in forge.commands:
        allowed = ", ".join(sorted(forge.commands))
        raise ToolError(
            f"`{argv[0]}` is not one of the {forge.label} commands this tool "
            f"offers. Available: {allowed}."
        )

    for token in argv[1:]:
        if token.lower() in WRITES:
            raise ToolError(
                f"`{token}` would change something on {forge.label}, and this "
                f"tool only reads. Use the shell tool if you mean to, so the "
                f"user is asked first."
            )
        if token in WRITE_FLAGS:
            raise ToolError(
                f"`{token}` can turn this into a write, and this tool only "
                f"reads. Use the shell tool if you mean to."
            )

    if argv[0] == "api":
        _check_api(argv, forge)
    return argv


def _check_api(argv: list[str], forge: Forge) -> None:
    """`api` is the escape hatch, so it gets looked at properly."""
    for index, token in enumerate(argv):
        if token in ("--method", "-X") and index + 1 < len(argv):
            if argv[index + 1].upper() not in READ_METHODS:
                raise ToolError(
                    f"only {' and '.join(sorted(READ_METHODS))} requests are "
                    f"allowed through `{forge.name} api`."
                )


def run(forge: Forge, command: str) -> ToolResult:
    """Run one read-only command against ``forge``."""
    argv = check(forge, command)
    if not shutil.which(forge.name):
        raise ToolError(f"`{forge.name}` is not installed.")

    try:
        done = subprocess.run(
            [forge.name, *argv],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"`{forge.name} {command}` took longer than {TIMEOUT}s.") from None
    except OSError as exc:
        raise ToolError(f"could not run {forge.name}: {exc}") from None

    out = (done.stdout or "").rstrip()
    err = (done.stderr or "").rstrip()

    if done.returncode != 0:
        # Not being logged in is the one failure worth naming, because the fix
        # is a command the user runs and the model never can.
        lowered = err.lower()
        if "auth" in lowered and ("login" in lowered or "token" in lowered):
            raise ToolError(
                f"{forge.label} is not authenticated here. The user needs to run "
                f"`{forge.login}`; you cannot do it for them."
            )
        # Everything else is information: a run that does not exist, a branch
        # with no pull request. The model usually needs to read it.
        body = "\n".join(part for part in (out, err) if part)
        return ToolResult.success(truncate(f"exit code {done.returncode}\n{body}".rstrip()))

    body = out or err
    return ToolResult.success(truncate(body) if body else "(no output)")


def register(registry: Registry, forge: Forge) -> Registry:
    """Add ``forge`` to the registry, if its CLI is installed.

    Absent, nothing is registered: a tool the model can see and cannot use is a
    turn spent finding that out.
    """
    if not shutil.which(forge.name):
        return registry

    registry.add(
        Tool(
            name=forge.name,
            description=(
                f"Read from {forge.label} with the `{forge.name}` CLI, which is "
                f"installed and already authenticated. Use this for anything "
                f"that is not a file in the workspace: CI runs and their logs, "
                f"pull requests, issues, releases. It only reads.\n"
                f"{forge.examples}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            f"The arguments to `{forge.name}`, without the "
                            f"program name. For example: run view 123 --log-failed"
                        ),
                    },
                },
                "required": ["command"],
            },
            run=lambda command: run(forge, command),
            mutating=False,
        )
    )
    return registry
