"""What git can be asked about the directory the session is in.

One question so far -- which branch this is -- asked for the row under the
prompt. It lives here rather than in the row that draws it because a status
line that shells out is a status line that has to know what a detached HEAD is,
and that is not a drawing concern.

Everything here answers ``""`` rather than raising. There may be no git, no
repository, or a repository git refuses to read; none of those is a reason for
the prompt not to appear, and a blank field is the whole of the report.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: How long git gets. The row is rebuilt whenever a line is submitted, so this
#: sits between the user pressing Enter and the turn starting -- long enough for
#: a cold filesystem, short enough that a hung git is a blink and not a hang.
TIMEOUT = 2.0

#: What git calls a HEAD that is not on a branch. Shown as the short commit
#: instead, because "HEAD" names every detached checkout there has ever been.
DETACHED = "HEAD"


def _run(args: list[str], root: Path | str | None, timeout: float) -> str:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Windows would otherwise flash a console window for each call.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def branch(root: Path | str | None = None, timeout: float = TIMEOUT) -> str:
    """The branch checked out at ``root``, or ``""`` when there is not one.

    A detached HEAD answers with the short commit it is sitting on. Reporting
    the literal ``HEAD`` git hands back would be true of every detached
    checkout and so tell nobody which one this is.
    """
    name = _run(["rev-parse", "--abbrev-ref", "HEAD"], root, timeout)
    if name and name != DETACHED:
        return name
    if name == DETACHED:
        return _run(["rev-parse", "--short", "HEAD"], root, timeout)
    return ""
