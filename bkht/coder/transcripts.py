"""Reading sessions, including other agents'.

Several coding agents run on the same machine and each keeps its transcripts in
a format of its own. They answer a question `coder sessions` could not: what was
already tried here, by whatever was open at the time. So the three that write
JSONL are read into one shape and listed together.

Read-only, and human-facing only. The model is given no tool for this and the
system prompt says nothing about it: context is the scarce thing in a local
session, and another agent's history is not what it should be spent on. If
something in an old session matters, the person reading it can say so.

Each reader is defensive to the point of dullness. These files belong to other
programs, which will change them without telling anybody, so every record is
tried and skipped rather than trusted -- a format that has moved on should cost
its own sessions, not the listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .parsing import strip_json
from .session import SESSION_DIR

CODER = "coder"
CLAUDE = "claude"
CODEX = "codex"
SOURCES = (CODER, CLAUDE, CODEX)

CLAUDE_DIR = Path("~/.claude/projects").expanduser()
CODEX_DIR = Path("~/.codex/sessions").expanduser()

#: How much of a message to keep. These are for reading, not for replaying into
#: a model, and a tool result of forty thousand characters is not read by anyone.
EXCERPT = 2000


@dataclass
class Turn:
    """One thing somebody or something said."""

    role: str
    text: str
    when: float | None = None


@dataclass
class Transcript:
    """One session, from whichever agent wrote it."""

    source: str
    id: str
    path: Path
    cwd: str = ""
    started: float | None = None
    title: str = ""
    short: str = ""
    opening: str = ""
    turns: list[Turn] = field(default_factory=list)

    @property
    def label(self) -> str:
        """`claude/1be46299`, short enough to type and still unambiguous.

        The distinctive part of an id is not in the same place in every format:
        a uuid is distinctive from its first character, and coder's ids begin
        with a date, so the first eight of those are the same for everything
        written that day.
        """
        return f"{self.source}/{self.short or self.id[:8]}"


def _head(path: Path, lines: int):
    """The first ``lines`` records of a file, for a listing.

    Reading every message of twelve hundred transcripts to print a table of
    twelve hundred rows takes five seconds and answers nothing the header does
    not. Every format puts its metadata at the top.
    """
    taken = 0
    for record in _records(path):
        yield record
        taken += 1
        if taken >= lines:
            return


def _records(path: Path):
    """Every JSON object in a JSONL file, skipping what will not parse.

    A killed session leaves a half-written last line, and these files are being
    appended to by another process while this reads them.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _moment(value) -> float | None:
    """A timestamp from whatever the format used to write one."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _text(content) -> str:
    """The prose in a message, whether it is a string or a block list.

    Tool calls, tool results and thinking are dropped. What is wanted is the
    conversation -- what was asked and what was answered -- and the rest is a
    log of how, which is longer than the whole of the transcript worth reading.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _turn(role: str, text: str, when=None) -> Turn | None:
    return Turn(role=role, text=text[:EXCERPT], when=_moment(when)) if text else None


# --- coder --------------------------------------------------------------------


def read_coder(path: Path) -> Transcript | None:
    """This package's own sessions: a header line, then one line per message."""
    made = Transcript(source=CODER, id=path.stem, path=path)
    made.short = path.stem.rsplit("-", 1)[-1]
    for record in _records(path):
        kind = record.get("type")
        if kind == "session":
            made.id = record.get("id") or made.id
            made.short = made.id.rsplit("-", 1)[-1]
            made.cwd = record.get("cwd") or ""
            made.started = _moment(record.get("created"))
        elif kind == "clear":
            made.turns.clear()
        elif kind == "message":
            role = record.get("role") or "user"
            if role == "tool":
                continue
            # An assistant message carries its tool call as ordinary content on
            # this model. On screen the renderer takes that out; a transcript
            # read back with it in is mostly JSON, so it gets the same
            # treatment, and a turn that was only a call drops out entirely.
            content = str(record.get("content") or "")
            if role == "assistant":
                content = strip_json(content)
            if turn := _turn(role, content):
                made.turns.append(turn)
    return made


# --- claude code --------------------------------------------------------------


def _claude_branch(records: list[dict]) -> list[dict]:
    """The records on the active branch, newest leaf backwards.

    Claude Code's file is a tree, not a list: `parentUuid` and `uuid` chain the
    records, and a rewind or an edit starts a sibling branch beside the one it
    replaced. Replaying the file in order therefore shows both, which reads as
    the same work done twice. The `last-prompt` record names the live leaf, so
    the branch is walked back from there.
    """
    by_uuid = {r["uuid"]: r for r in records if isinstance(r.get("uuid"), str)}
    leaf = next(
        (r.get("leafUuid") for r in reversed(records) if r.get("type") == "last-prompt"),
        None,
    )
    if leaf not in by_uuid:
        return records

    chain: list[dict] = []
    seen: set[str] = set()
    node = leaf
    while node in by_uuid and node not in seen:
        seen.add(node)
        chain.append(by_uuid[node])
        node = by_uuid[node].get("parentUuid")
    chain.reverse()
    return chain


def read_claude(path: Path) -> Transcript | None:
    records = list(_records(path))
    if not records:
        return None

    made = Transcript(source=CLAUDE, id=path.stem, path=path, short=path.stem[:8])
    for record in records:
        made.cwd = made.cwd or (record.get("cwd") or "")
        made.title = made.title or (record.get("aiTitle") or "")
        if made.started is None:
            made.started = _moment(record.get("timestamp"))

    for record in _claude_branch(records):
        kind = record.get("type")
        if kind not in ("user", "assistant"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        # A user record whose content is a block list is a tool result being
        # fed back, not a person typing. Only the plain strings were typed.
        content = message.get("content")
        if kind == "user" and not isinstance(content, str):
            continue
        if turn := _turn(kind, _text(content), record.get("timestamp")):
            made.turns.append(turn)
    return made


def claude_sessions() -> list[Path]:
    return sorted(CLAUDE_DIR.glob("*/*.jsonl")) if CLAUDE_DIR.is_dir() else []


# --- codex --------------------------------------------------------------------


def read_codex(path: Path) -> Transcript | None:
    made = Transcript(source=CODEX, id=path.stem, path=path, short=path.stem[-36:][:8])
    for record in _records(path):
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")

        if record.get("type") == "session_meta":
            made.id = payload.get("session_id") or payload.get("id") or made.id
            made.short = str(made.id)[:8]
            made.cwd = payload.get("cwd") or ""
            made.started = _moment(record.get("timestamp"))
            continue

        # `response_item` and `event_msg` are two views of the same turn, so
        # reading both would show every message twice. The event view is the
        # one written for a reader.
        if record.get("type") != "event_msg":
            continue
        if kind == "user_message":
            turn = _turn("user", str(payload.get("message") or ""), record.get("timestamp"))
        elif kind == "agent_message":
            turn = _turn("assistant", str(payload.get("message") or ""), record.get("timestamp"))
        else:
            continue
        if turn:
            made.turns.append(turn)
    return made


def codex_sessions() -> list[Path]:
    return sorted(CODEX_DIR.glob("*/*/*/rollout-*.jsonl")) if CODEX_DIR.is_dir() else []


# --- listing ------------------------------------------------------------------
#
# A header is the metadata plus the first thing the user said, which is what a
# listing shows. Read from the top of the file and stopped early, so listing
# every session on the machine costs a few lines each rather than all of them.

HEAD_LINES = 60


def head_coder(path: Path) -> Transcript | None:
    made = Transcript(source=CODER, id=path.stem, path=path)
    made.short = path.stem.rsplit("-", 1)[-1]
    for record in _head(path, HEAD_LINES):
        if record.get("type") == "session":
            made.id = record.get("id") or made.id
            made.short = made.id.rsplit("-", 1)[-1]
            made.cwd = record.get("cwd") or ""
            made.started = _moment(record.get("created"))
        elif record.get("type") == "message" and record.get("role") == "user":
            made.opening = str(record.get("content") or "")[:200]
            break
    return made


def head_claude(path: Path) -> Transcript | None:
    made = Transcript(source=CLAUDE, id=path.stem, path=path, short=path.stem[:8])
    for record in _head(path, HEAD_LINES):
        made.cwd = made.cwd or (record.get("cwd") or "")
        made.title = made.title or (record.get("aiTitle") or "")
        if made.started is None:
            made.started = _moment(record.get("timestamp"))
        message = record.get("message")
        if record.get("type") == "user" and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip() and not made.opening:
                made.opening = content.strip()[:200]
    return made


def head_codex(path: Path) -> Transcript | None:
    made = Transcript(source=CODEX, id=path.stem, path=path, short=path.stem[-36:][:8])
    for record in _head(path, HEAD_LINES):
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "session_meta":
            made.id = payload.get("session_id") or payload.get("id") or made.id
            made.short = str(made.id)[:8]
            made.cwd = payload.get("cwd") or ""
            made.started = _moment(record.get("timestamp"))
        elif payload.get("type") == "user_message" and not made.opening:
            made.opening = str(payload.get("message") or "").strip()[:200]
            break
    return made


HEADS = {CODER: head_coder, CLAUDE: head_claude, CODEX: head_codex}


# --- everything together ------------------------------------------------------

READERS = {CODER: read_coder, CLAUDE: read_claude, CODEX: read_codex}


def _paths(source: str) -> list[Path]:
    if source == CODER:
        return sorted(SESSION_DIR.glob("*.jsonl")) if SESSION_DIR.is_dir() else []
    if source == CLAUDE:
        return claude_sessions()
    return codex_sessions()


def load(path: Path, source: str, whole: bool = True) -> Transcript | None:
    """One transcript, or ``None`` when the file cannot be made sense of.

    ``whole=False`` reads only the header, which is all a listing needs.
    """
    try:
        return (READERS if whole else HEADS)[source](Path(path))
    except Exception:
        # These files belong to other programs and will change without notice.
        # A format that has moved on costs its own session, not the listing.
        return None


def discover(root: Path | str | None = None, sources=SOURCES) -> list[Transcript]:
    """Every readable session, newest first, optionally for one directory only.

    Matched on the `cwd` recorded inside the file rather than on the directory
    name: Claude Code flattens a path into its directory name by replacing both
    slashes and spaces with dashes, which two different paths can share.
    """
    wanted = str(Path(root).expanduser().resolve()) if root else None
    found: list[Transcript] = []
    for source in sources:
        for path in _paths(source):
            made = load(path, source, whole=False)
            if made is None or not made.opening:
                continue
            if wanted and made.cwd and str(Path(made.cwd)) != wanted:
                continue
            found.append(made)
    found.sort(key=lambda t: (t.started or 0.0), reverse=True)
    return found


def find(label: str, root: Path | str | None = None) -> Transcript | None:
    """A transcript by `source/id`, by id, or by an unambiguous id prefix."""
    source, _, wanted = label.partition("/")
    if source not in SOURCES:
        source, wanted = "", label
    matches = [
        made
        for made in discover(root=None, sources=(source,) if source else SOURCES)
        if wanted in (made.id, made.short) or made.id.startswith(wanted)
    ]
    # An exact id beats every prefix that merely starts with it.
    exact = [made for made in matches if wanted in (made.id, made.short)]
    if len(exact) == 1:
        return load(exact[0].path, exact[0].source)
    if len(matches) == 1:
        return load(matches[0].path, matches[0].source)
    return None


def when(moment: float | None) -> str:
    if not moment:
        return "unknown"
    return datetime.fromtimestamp(moment, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
