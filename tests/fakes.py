"""A scripted provider, so loop logic is tested without a model."""

from __future__ import annotations

from typing import Iterator

from bkht.coder.provider import Chunk, ProviderError


class FakeProvider:
    """Replays a scripted list of replies, one per ``chat`` call.

    Each script entry is either a string (streamed as content) or an exception
    to raise. Replies are chunked mid-string so the tests exercise the same
    reassembly path the real stream uses.
    """

    def __init__(self, script: list, model: str = "fake") -> None:
        self.script = list(script)
        self.model = model
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[Chunk]:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("FakeProvider ran out of scripted replies")

        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply

        for i in range(0, len(reply), 7):
            yield Chunk(content=reply[i : i + 7])
        yield Chunk(done=True, prompt_tokens=100, completion_tokens=len(reply))

    @property
    def exhausted(self) -> bool:
        return not self.script


def call(name: str, **arguments) -> str:
    """A scripted tool call in the wire format the model actually emits."""
    import json

    return json.dumps({"name": name, "arguments": arguments})
