"""The ``coder sessions`` and ``coder session`` commands.

Every session is already a durable file -- append-only JSONL, written as the
turn happens -- but until now the only way to reach one was ``--resume``, which
silently picks the newest for the current directory. That is the right default
and the wrong only option: a day's work leaves a dozen transcripts, and the one
worth continuing is rarely the last one.

Listing and inspecting live here rather than in ``session.py`` for the same
reason ``review/cli.py`` is not ``review/reviewer.py``: the store should not
know what a terminal is.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .session import Session, find, sessions_for
from .terminal import DIM, YELLOW, paint

#: Transcript lines shown by `coder session <id>` before the rest is summarised.
PREVIEW = 20

LAST = "last"


def ago(created: float, now: float | None = None) -> str:
    """How long ago ``created`` was, in the coarsest unit that still says it.

    Coarse on purpose: the question a listing answers is "which of these is the
    one I was just in", and to the minute is more than that needs.
    """
    if not created:
        return "unknown"
    seconds = (time.time() if now is None else now) - created
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 172800:
        return "yesterday"
    if seconds < 2592000:
        return f"{int(seconds // 86400)}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(created))


def _payload(info) -> dict:
    return {
        "id": info.id,
        "path": str(info.path),
        "cwd": info.cwd,
        "model": info.model,
        "created": info.created,
        "messages": info.messages,
    }


def report(root: Path, *, as_json: bool = False, everywhere: bool = False,
           out=print) -> int:
    """List saved sessions: this workspace's, or every one on the machine."""
    found = sessions_for(None if everywhere else str(root))

    if as_json:
        out(json.dumps([_payload(info) for info in found], indent=2))
        return 0

    if not found:
        scope = "on this machine" if everywhere else f"for {root}"
        out(f"No saved sessions {scope}.")
        return 0

    for info in found:
        line = f"  {info.id}  {ago(info.created):<11}{info.model or '?':<22}{info.messages} msgs"
        if everywhere:
            line = f"{line}  {info.cwd}"
        out(line)
    out(paint(f"\n{len(found)} session{'s' if len(found) != 1 else ''}. "
              f"Resume one with `coder session resume <id>`.", DIM))
    return 0


def resolve(root: Path, target: str) -> Path | None:
    """The file a `last`-or-id argument names, or None.

    ``last`` is scoped to the workspace and an id is not: an id is already
    unambiguous, and refusing to open one because it was recorded in another
    directory would just be a second thing to explain.
    """
    if not target or target == LAST:
        return Session.latest_for(str(root))
    return find(target)


def missing(root: Path, target: str) -> str:
    """What to say when :func:`resolve` found nothing."""
    if not target or target == LAST:
        return f"No saved sessions for {root}."
    return f"No session matches {target!r}. Run `coder sessions` to see what there is."


def _preview(message: dict) -> str:
    role = message.get("role", "?")
    body = " ".join(str(message.get("content", "")).split())
    if role == "tool":
        body = f"{message.get('name', '?')}: {body}"
    if len(body) > 70:
        body = body[:67] + "..."
    return f"  {role:<11}{body}"


def show(root: Path, target: str, *, as_json: bool = False, out=print) -> int:
    """Print one session: its header facts, then a skim of the transcript."""
    path = resolve(root, target)
    if path is None:
        print(paint(missing(root, target), YELLOW, sys.stderr), file=sys.stderr)
        return 1

    session = Session.load(path)
    if as_json:
        out(json.dumps({
            "id": session.id,
            "path": str(path),
            "cwd": session.cwd,
            "model": session.model,
            "messages": session.messages,
        }, indent=2))
        return 0

    out(f"{session.id} · {session.model or '?'} · {len(session.messages)} messages")
    out(paint(f"{session.cwd}\n{path}", DIM))

    if not session.messages:
        out("\nNothing was said in this session.")
        return 0

    # The tail, not the head: what you want before resuming is where it got to.
    hidden = len(session.messages) - PREVIEW
    out("")
    if hidden > 0:
        out(paint(f"  ... {hidden} earlier message{'s' if hidden != 1 else ''}", DIM))
    for message in session.messages[-PREVIEW:]:
        out(_preview(message))
    return 0
