"""Tolerant extraction of JSON out of model message content.

Two features depend on this: tool calls (the model emits them as ordinary
content, not in ``message.tool_calls``) and review findings. The transport
problem is identical in both cases, so it is one parser, not two.

The scan is brace/bracket matching rather than a regex, because a regex cannot
handle nesting, and small models routinely wrap their JSON in prose or fences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

OPENERS = {"{": "}", "[": "]"}


def scan_balanced(text: str, start: int) -> int | None:
    """Return the index just past the balanced value opening at ``start``.

    ``None`` if the value never closes. String contents and escapes are
    honoured so that braces inside strings do not affect the depth count.
    """
    closer = OPENERS[text[start]]
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in OPENERS:
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1
            if depth == 0:
                return i + 1 if ch == closer else None

    return None


def extract_json_values(text: str) -> list[Any]:
    """Extract every top-level JSON object/array embedded in ``text``.

    Handles bare JSON, fenced blocks, JSON surrounded by prose, and several
    values in one message. Fragments that do not parse are skipped silently --
    the caller decides what an empty result means.
    """
    if not text:
        return []

    values: list[Any] = []
    i = 0
    while i < len(text):
        if text[i] not in OPENERS:
            i += 1
            continue

        end = scan_balanced(text, i)
        if end is None:
            i += 1
            continue

        try:
            values.append(json.loads(text[i:end]))
        except json.JSONDecodeError:
            i += 1
            continue
        i = end

    return values


def _flatten(values: list[Any]) -> list[dict]:
    """Flatten top-level arrays so ``[{...}, {...}]`` yields two dicts."""
    out: list[dict] = []
    for value in values:
        if isinstance(value, dict):
            out.append(value)
        elif isinstance(value, list):
            out.extend(item for item in value if isinstance(item, dict))
    return out


def extract_json_objects(text: str) -> list[dict]:
    """Every JSON object in ``text``, with top-level arrays flattened."""
    return _flatten(extract_json_values(text))


@dataclass
class ToolCall:
    """A single request from the model to run a tool.

    ``raw`` keeps the originating object so a schema violation can be reported
    back to the model in terms of what it actually emitted.
    """

    name: str
    arguments: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _as_arguments(value: Any) -> dict | None:
    """Coerce an ``arguments`` field, which is sometimes a JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = extract_json_objects(value)
        if len(parsed) == 1:
            return parsed[0]
    if value is None:
        return {}
    return None


def _as_tool_call(obj: dict) -> ToolCall | None:
    """Read one tool call out of a JSON object, tolerating known shapes."""
    # Ollama's native shape, in case a future model populates tool_calls.
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]

    name = obj.get("name") or obj.get("tool")
    if not isinstance(name, str) or not name:
        return None

    raw_args = obj.get("arguments")
    if raw_args is None:
        raw_args = obj.get("parameters")
    if raw_args is None:
        raw_args = obj.get("args")

    arguments = _as_arguments(raw_args)
    if arguments is None:
        return None

    return ToolCall(name=name, arguments=arguments, raw=obj)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Every tool call the model emitted in message content, in order."""
    calls = []
    for obj in extract_json_objects(text):
        call = _as_tool_call(obj)
        if call is not None:
            calls.append(call)
    return calls


def is_tool_call(text: str) -> bool:
    """True when ``text`` is one tool call rather than prose.

    The question every caller actually has, asked once. Testing the first
    character for `{` answers a different one: the model routinely wraps its
    call in ```json ... ```, and a fence begins with a backtick -- so the
    prefix test reads a call as prose and every check that follows is skipped.

    One call, not "at least one": content carrying two of them round-trips to
    something other than what it was rendered from, which is precisely what the
    callers here exist to refuse.
    """
    return len(parse_tool_calls(text)) == 1


#: A code fence, as the model writes one around a tool call it is about to
#: make. Three backticks or more, optionally with a language after them.
FENCE = "```"


def _fence_lines(text: str) -> list[tuple[int, int]]:
    """Every fence line in ``text``, as ``(start, end)`` character offsets."""
    found = []
    start = 0
    for line in text.splitlines(keepends=True):
        if line.strip().startswith(FENCE):
            found.append((start, start + len(line)))
        start += len(line)
    return found


def open_fence(text: str) -> int | None:
    """Where an unclosed code fence begins, if one does.

    Streaming has to hold a fence back until its contents are known: the
    opening line arrives whole, several chunks before whatever it contains, so
    by the time a tool call inside it has been recognised and removed the fence
    is already on the screen and cannot be taken off it.
    """
    fences = _fence_lines(text)
    return None if len(fences) % 2 == 0 else fences[-1][0]


def drop_empty_fences(text: str) -> str:
    """``text`` with fences that now contain nothing at all removed.

    The model wraps its tool call in ```json ... ```. Removing the call leaves
    the fence standing around nothing, which renders as an empty block -- the
    blank gap that used to sit above every tool call in a transcript.
    """
    fences = _fence_lines(text)
    if len(fences) < 2:
        return text
    out = []
    cut = 0
    for opener, closer in zip(fences[::2], fences[1::2]):
        if text[opener[1] : closer[0]].strip():
            continue
        out.append(text[cut : opener[0]])
        cut = closer[1]
    out.append(text[cut:])
    return "".join(out)


def strip_json(text: str) -> str:
    """``text`` with every extracted JSON value removed.

    Used to decide whether a reply carrying a tool call also carried prose
    worth showing the user.
    """
    if not text:
        return ""

    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in OPENERS:
            end = scan_balanced(text, i)
            if end is not None:
                try:
                    json.loads(text[i:end])
                except json.JSONDecodeError:
                    pass
                else:
                    i = end
                    continue
        out.append(text[i])
        i += 1

    return drop_empty_fences("".join(out)).strip()
