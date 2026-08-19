"""Message history.

Persistence and compaction arrive in later steps; for now this is the single
place that owns the message list, so nothing else has to know its shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
