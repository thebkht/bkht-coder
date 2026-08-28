"""Message history, persistence, and undo snapshots.

Sessions are append-only JSONL under ``~/.bkht-coder/sessions/``: one header
record naming the working directory and model, then one record per message.
Append-only means a crashed or killed session is still resumable up to its last
complete message, which matters when a turn can take minutes.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .language import ENGLISH

STATE_DIR = Path("~/.bkht-coder").expanduser()
SESSION_DIR = STATE_DIR / "sessions"


def new_session_id() -> str:
    """A sortable id, so 'most recent' is a filename comparison."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass
class Session:
    """The conversation with the model, optionally backed by a JSONL file."""

    system: str = ""
    messages: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    id: str = field(default_factory=new_session_id)
    cwd: str = ""
    model: str = ""
    path: Path | None = None
    # The language the user is writing in, carried between turns because a
    # follow-up is often too short to identify on its own. Derived state, not
    # transcript: it is never written to the file.
    language: str | None = None

    # --- history ------------------------------------------------------------

    def payload(self) -> list[dict]:
        """The full message list to send, system prompt first.

        When the user is not writing in English the list ends with a reminder
        naming their language. It is built here, on every request, and never
        appended to ``messages``: a reminder in the history would pile up one
        copy per turn, survive into a resumed session, and be swept into the
        next summary. Regenerating it keeps exactly one, always last.
        """
        head = [{"role": "system", "content": self.system}] if self.system else []
        return head + self.messages + self._reminders()

    def _reminders(self) -> list[dict]:
        """The ephemeral messages appended to the end of every request.

        Sent as ``user`` rather than ``system``. A fresh system-role message
        arriving as the very last thing before generation reads, to a small
        model, like the request it is supposed to answer -- so it answers *it*,
        greeting the user in the named language instead of doing the work. As a
        user turn it is plainly an aside attached to the conversation, in the
        one position where it will actually be read.
        """
        # English needs no reminder: the prompt is already written in it, and
        # that is the common case, so the common case costs nothing.
        if not self.language or self.language == ENGLISH:
            return []
        content = prompts.language_reminder(self.language)
        return [{"role": "user", "content": content}]

    def add_user(self, content: str, images: list[str] | None = None) -> None:
        """Record what the user said, and any images they pasted with it.

        The paths travel on the message rather than in its text, so a provider
        that can send pictures sends them and one that cannot is not handed a
        wall of base64 it will try to read as prose. They are paths, not bytes:
        a transcript is a file somebody may open, and the bytes are already on
        disk under the state directory.
        """
        message = {"role": "user", "content": content}
        if images:
            message["images"] = list(images)
        self._append(message)

    def add_assistant(self, content: str) -> None:
        self._append({"role": "assistant", "content": content})

    def add_tool_result(self, name: str, content: str) -> None:
        self._append({"role": "tool", "name": name, "content": content})

    def _append(self, message: dict) -> None:
        self.messages.append(message)
        self._persist({"type": "message", **message})

    def clear(self) -> None:
        """Drop the history, keeping the system prompt and the file."""
        self.messages.clear()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._persist({"type": "clear"})

    def record_usage(self, prompt: int | None, completion: int | None) -> None:
        """Track token counts so context pressure can be measured."""
        if prompt:
            self.prompt_tokens = prompt
        if completion:
            self.completion_tokens += completion

    # --- persistence --------------------------------------------------------

    def start_file(self, directory: Path | None = None) -> Path:
        """Begin persisting this session, writing the header record."""
        directory = Path(directory) if directory else SESSION_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.id}.jsonl"
        self._persist(
            {
                "type": "session",
                "id": self.id,
                "cwd": self.cwd,
                "model": self.model,
                "created": time.time(),
            }
        )
        for message in self.messages:
            self._persist({"type": "message", **message})
        return self.path

    def _persist(self, record: dict) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            # Losing the transcript must never take the session down with it.
            self.path = None

    @classmethod
    def load(cls, path: Path, system: str = "") -> "Session":
        """Rebuild a session from its JSONL file.

        Malformed lines are skipped rather than fatal: the file is append-only
        and a session killed mid-write can leave a partial last line.
        """
        session = cls(system=system, path=Path(path))
        session.messages = []

        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = record.get("type")
            if kind == "session":
                session.id = record.get("id", session.id)
                session.cwd = record.get("cwd", "")
                session.model = record.get("model", "")
            elif kind == "clear":
                session.messages = []
            elif kind == "message":
                message = {k: v for k, v in record.items() if k != "type"}
                if message.get("role"):
                    session.messages.append(message)

        return session

    @staticmethod
    def latest_for(cwd: str, directory: Path | None = None) -> Path | None:
        """The most recent session file recorded for ``cwd``, if any."""
        for path, _ in headers(cwd, directory):
            return path
        return None


@dataclass(frozen=True)
class Info:
    """One saved session, as it appears in a listing."""

    id: str
    path: Path
    cwd: str
    model: str
    created: float
    messages: int


def headers(cwd: str | None = None, directory: Path | None = None):
    """Every saved session as ``(path, header)``, newest first.

    Only the first line of each file is read, which is what makes listing
    cheap: the header carries the directory, the model and the creation time,
    and a session that is still being appended to answers just as fast as one
    that finished last week. ``cwd`` of None means every workspace.

    Files that cannot be read, or whose first line is not a header, are
    skipped: the directory is the user's, and a stray file in it is not a
    reason to refuse to list the real ones.
    """
    directory = Path(directory) if directory else SESSION_DIR
    if not directory.is_dir():
        return

    target = os.path.realpath(cwd) if cwd is not None else None
    # Ids are timestamp-prefixed, so newest-first is a reverse name sort.
    for path in sorted(directory.glob("*.jsonl"), reverse=True):
        try:
            with path.open(encoding="utf-8") as handle:
                header = json.loads(handle.readline() or "{}")
        except (OSError, json.JSONDecodeError):
            continue
        if header.get("type") != "session":
            continue
        if target is not None and header.get("cwd") != target:
            continue
        yield path, header


def sessions_for(cwd: str | None = None, directory: Path | None = None) -> list[Info]:
    """Saved sessions, newest first, with their message counts.

    Counting means reading each file, so this is the expensive listing and
    :func:`headers` is the cheap one. It is worth it: a list of ids and times
    with no sense of how much was said in each is a list you have to open one
    by one to use.
    """
    found = []
    for path, header in headers(cwd, directory):
        try:
            messages = len(Session.load(path).messages)
        except OSError:
            continue
        found.append(
            Info(
                id=header.get("id", path.stem),
                path=path,
                cwd=header.get("cwd", ""),
                model=header.get("model", ""),
                created=header.get("created", 0.0),
                messages=messages,
            )
        )
    return found


def find(session_id: str, directory: Path | None = None) -> Path | None:
    """The file for ``session_id``: an exact id, or an unambiguous prefix.

    Prefixes exist because the ids are long and timestamped, and the first
    eight characters are already a date. An ambiguous prefix returns None
    rather than the newest match -- resuming the wrong conversation is worse
    than being asked to type more of the id.
    """
    matches = []
    for path, header in headers(None, directory):
        identifier = header.get("id", path.stem)
        if identifier == session_id:
            return path
        if identifier.startswith(session_id):
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


@dataclass
class Snapshots:
    """Previous contents of every file a mutating tool touched.

    Undo is built on snapshots rather than git so it works in a directory that
    is not a repository -- which is exactly where letting a weak model write
    files is most alarming. ``None`` records a file that did not exist yet, so
    undoing a creation deletes it.
    """

    entries: list[tuple[Path, str | None]] = field(default_factory=list)

    def capture(self, path: Path) -> None:
        """Record ``path`` as it is now, before it is changed."""
        if path.exists() and path.is_file():
            try:
                self.entries.append((path, path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                self.entries.append((path, None))
        else:
            self.entries.append((path, None))

    def undo(self) -> str | None:
        """Restore the most recent snapshot. Returns what it did, or None."""
        if not self.entries:
            return None

        path, before = self.entries.pop()
        if before is None:
            if path.exists():
                path.unlink()
                return f"removed {path.name}"
            return f"{path.name} was already absent"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(before, encoding="utf-8")
        return f"restored {path.name}"

    def __len__(self) -> int:
        return len(self.entries)
