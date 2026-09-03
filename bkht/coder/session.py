"""Message history, persistence, and undo snapshots.

Sessions are append-only JSONL under ``~/.bkht-coder/sessions/``: one header
record naming the working directory and model, then one record per message.
Append-only means a crashed or killed session is still resumable up to its last
complete message, which matters when a turn can take minutes.

Three of the record types are written for a reader that is not the resume path.
A transcript is only a training example if the *input* half survives too, and
the input half is the system prompt: the tool protocol lives in it, and it is
assembled per session from the registry, the tree, the project instructions and
the skills. So ``prompt`` records what the model was actually told, ``outcome``
records how each turn ended, and the header names the backend that answered.
None of the three is replayed on resume -- a resumed session is rebuilt against
the tools it has *now*, which is the whole reason the old prompt has to be
written down rather than recomputed later.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .language import ENGLISH
from .plan import Plan

STATE_DIR = Path("~/.bkht-coder").expanduser()
SESSION_DIR = STATE_DIR / "sessions"


def _version() -> str:
    """Which coder wrote this file.

    Imported inside the call because ``doctor`` reaches for distribution
    metadata, and opening a session file is not worth paying that at import
    time. A build that cannot say what it is records nothing rather than a
    guess.
    """
    try:
        from .doctor import version

        return version()
    except Exception:
        return ""


def prompt_hash(system: str) -> str:
    """A short, stable name for one system prompt.

    Twelve hex characters, which is plenty to tell two prompts apart in a
    listing and short enough to read. Nothing security-sensitive rides on it.
    """
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]


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
    #: The backend that answered. Recorded because a trajectory's worth as
    #: training data depends entirely on what produced it: a frontier model
    #: through `claude-code` is a teacher, and the local 14b is a student.
    provider: str = ""
    #: The tool names offered this session, which depend on the mode and on the
    #: skills/planning/delegation switches -- and, for the shell, on the OS.
    tools: list[str] = field(default_factory=list)
    path: Path | None = None
    # The language the user is writing in, carried between turns because a
    # follow-up is often too short to identify on its own. Derived state, not
    # transcript: it is never written to the file.
    language: str | None = None
    # What the turn said it was going to do. Beside the messages rather than in
    # them, which is the whole point: compaction and elision free space by
    # dropping message history, and this is the one thing a turn cannot afford
    # to lose to them. Persisted, so a resumed session resumes the plan too.
    plan: Plan = field(default_factory=Plan)
    #: What the file said about itself, for a reader that wants the session as
    #: it was rather than as it would be rebuilt. Empty for a live session --
    #: nothing has been read back -- and never consulted by the resume path.
    recorded: dict = field(default_factory=dict)

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
        notes = []
        # English needs no reminder: the prompt is already written in it, and
        # that is the common case, so the common case costs nothing.
        if self.language and self.language != ENGLISH:
            notes.append(prompts.language_reminder(self.language))
        # Last, so the plan is the final thing read before the reply. An empty
        # plan says nothing at all: a turn that needed no plan should not pay
        # for the words explaining that it has none.
        if self.plan:
            notes.append(prompts.plan_reminder(self.plan.render()))
        return [{"role": "user", "content": note} for note in notes]

    def set_plan(self, steps: list[str]) -> None:
        """Replace the plan and record it, so a resumed session still has it."""
        self.plan.set(steps)
        self._persist_plan()

    def tick_plan(self, number: int):
        """Mark a step done and record it. Raises IndexError for a bad number."""
        step = self.plan.tick(number)
        self._persist_plan()
        return step

    def _persist_plan(self) -> None:
        self._persist({"type": "plan", "steps": self.plan.as_record()})

    def record_prompt(self, system: str, tools: list[str] | None = None) -> None:
        """Adopt a system prompt, and write down what it was.

        Called instead of assigning ``system`` directly, because the assignment
        is the moment the information exists: the prompt names the tool set, so
        it cannot be built until the registry is, which is after the file was
        opened. Written as its own record rather than into the header for the
        same reason -- and a resumed session appends a second one, so a file
        that spans two tool sets says so in order instead of claiming the newer
        one applied from the start.

        The hash is there so a reader can group trajectories by the prompt that
        produced them without comparing four kilobytes of text per session.
        """
        self.system = system
        self.tools = list(tools or [])
        self._persist(
            {
                "type": "prompt",
                "system": system,
                "hash": prompt_hash(system),
                "tools": self.tools,
                "provider": self.provider,
                "model": self.model,
            }
        )

    def record_outcome(self, outcome) -> None:
        """Write down how a turn ended.

        A transcript says what was said; this says whether it went anywhere. The
        difference matters to anything selecting trajectories to learn from: a
        turn that hit the iteration cap and a turn that answered look identical
        in the message list, and only one of them is worth imitating.

        Duck-typed rather than importing ``agent.Outcome``, which would make the
        session depend on the loop that drives it.
        """
        self._persist(
            {
                "type": "outcome",
                "stopped": getattr(outcome, "stopped", ""),
                "iterations": getattr(outcome, "iterations", 0),
                "tool_calls": getattr(outcome, "tool_calls", 0),
                "errors": list(getattr(outcome, "errors", []) or []),
                "seconds": round(float(getattr(outcome, "seconds", 0.0)), 3),
                "sent": getattr(outcome, "sent", 0),
                "received": getattr(outcome, "received", 0),
            }
        )

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
        # The plan described the conversation being dropped. Surviving it would
        # mean the next turn opens against a checklist for work nobody asked
        # for -- exactly the confusion the plan exists to prevent.
        self.plan = Plan()
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
                "provider": self.provider,
                "version": _version(),
                "created": time.time(),
            }
        )
        # A prompt already assigned before the file existed is written down
        # too, so the ordering rule -- every message follows the prompt in
        # force for it -- holds for a session whose file opened late.
        if self.system:
            self.record_prompt(self.system, self.tools)
        for message in self.messages:
            self._persist({"type": "message", **message})
        # A plan made before the file existed would otherwise be the one piece
        # of the session the file did not describe.
        if self.plan:
            self._persist_plan()
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
                session.provider = record.get("provider", "")
                session.recorded.update(record)
            elif kind in ("prompt", "outcome"):
                # Kept beside the session rather than in it. The prompt read
                # here described the tools of the run that wrote it, and this
                # run is about to be given its own -- so replaying it would
                # hand the model a protocol for tools it no longer has.
                session.recorded.setdefault(kind, []).append(record)
            elif kind == "clear":
                session.messages = []
                session.plan = Plan()
            elif kind == "plan":
                # Replayed rather than merged: the file is a log of every
                # version the plan had, and the last one is the plan.
                session.plan = Plan.from_record(record.get("steps"))
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
