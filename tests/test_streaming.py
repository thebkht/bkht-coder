"""The streaming gate: prose reaches the screen, tool calls do not."""

from __future__ import annotations

import pytest

from bkht.coder.parsing import strip_json
from bkht.coder.streaming import Gate, visible

CALL = '{"name": "read_file", "arguments": {"path": "a.py"}}'

REPLIES = [
    "Just prose, no JSON at all.",
    CALL,
    f"I'll read the file.\n{CALL}",
    f"{CALL}\nDone.",
    f"Before {CALL} after {CALL} end.",
    'Here is a dict: {"a": 1} in prose.',
    f"```json\n{CALL}\n```",
    '{"unclosed": "this never finishes',
    "A brace { that is not JSON at all.",
    "",
    "[1, 2, 3] is an array.",
    'Nested {"a": {"b": [1, {"c": 2}]}} value.',
]


def collect(reply: str, chunks: list[str]) -> str:
    out: list[str] = []
    gate = Gate(out.append)
    for chunk in chunks:
        gate.feed(chunk)
    gate.finish()
    return "".join(out)


def split_every(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@pytest.mark.parametrize("reply", REPLIES, ids=lambda r: (r[:24] or "empty"))
def test_the_gate_shows_exactly_what_the_final_answer_keeps(reply):
    # The contract: streamed output and strip_json agree on what was prose.
    assert collect(reply, [reply]).strip() == strip_json(reply)


@pytest.mark.parametrize("reply", REPLIES, ids=lambda r: (r[:24] or "empty"))
@pytest.mark.parametrize("size", [1, 2, 3, 7, 100])
def test_chunking_never_changes_what_is_shown(reply, size):
    # Every split point must produce the same screen, or a tool call could leak
    # simply because the network happened to break mid-brace.
    assert collect(reply, split_every(reply, size)) == collect(reply, [reply])


@pytest.mark.parametrize("size", [1, 5])
def test_a_tool_call_never_reaches_the_screen(size):
    shown = collect(CALL, split_every(CALL, size))
    assert "read_file" not in shown
    assert shown.strip() == ""


def test_an_unclosed_brace_is_released_as_prose_at_the_end():
    text = 'Consider {"a": 1'
    out: list[str] = []
    gate = Gate(out.append)
    gate.feed(text)
    assert "".join(out) == "Consider ", "the open brace must be held while it could still close"
    gate.finish()
    assert "".join(out) == text


def test_prose_before_a_call_is_shown_immediately():
    out: list[str] = []
    gate = Gate(out.append)
    gate.feed("I'll read it. ")
    assert "".join(out) == "I'll read it. "
    gate.feed(CALL)
    assert "".join(out) == "I'll read it. "


def test_leading_blank_space_before_a_call_is_swallowed():
    out: list[str] = []
    gate = Gate(out.append)
    gate.feed(f"\n\n{CALL}")
    gate.finish()
    assert "".join(out) == ""


def test_visible_matches_strip_json_on_every_reply():
    for reply in REPLIES:
        assert visible(reply, final=True).strip() == strip_json(reply)
