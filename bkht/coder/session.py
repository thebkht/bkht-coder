"""Message history.

Persistence and compaction arrive in later steps; for now this is the single
place that owns the message list, so nothing else has to know its shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Session:
    """The conversation with the model."""

    system: str = ""
    messages: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def payload(self) -> list[dict]:
        """The full message list to send, system prompt first."""
        head = [{"role": "system", "content": self.system}] if self.system else []
        return head + self.messages

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, name: str, content: str) -> None:
        self.messages.append({"role": "tool", "name": name, "content": content})

    def record_usage(self, prompt: int | None, completion: int | None) -> None:
        """Track token counts so context pressure can be measured."""
        if prompt:
            self.prompt_tokens = prompt
        if completion:
            self.completion_tokens += completion


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
