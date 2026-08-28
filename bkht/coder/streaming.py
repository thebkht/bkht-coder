"""Show the model's prose as it arrives, without leaking its tool calls.

Tool calls travel as ordinary message content on this model, so streaming the
raw token feed puts naked JSON on the screen. Hiding it after the fact is not
an option either -- by then it has already been printed.

The contract this holds to, and the reason it is safe to withhold text
mid-stream: **the gate hides exactly what** :func:`parsing.strip_json` **removes
from the finished reply -- no more, no less.** So the streamed transcript and
the final answer never disagree about what was prose.

The trick that makes it chunk-independent is that a prefix is settled as soon
as no unclosed opener precedes it: an already-balanced value stays balanced,
and a character already judged prose stays prose. Only the tail from a
still-open brace is held back, and a brace that never closes was prose all
along, so it is released at the end of the turn.

An open code fence is held back for the same reason and needs its own rule,
because the fence is not a JSON opener. The model writes ```json around the
call it is about to make, and that whole line arrives several chunks before the
call inside it -- so a gate that settled it immediately had already printed a
fence it was about to empty. Held, the fence is only released once what is
inside it is known: with something in it, or not at all.
"""

from __future__ import annotations

import json
from typing import Callable

from .parsing import FENCE, OPENERS, drop_empty_fences, open_fence, scan_balanced


def _held(text: str) -> int | None:
    """Where a still-undecided code fence begins, if one does.

    Two ways a fence is undecided. It has opened and not yet closed, so what is
    inside it is still arriving. Or its own first line is still arriving --
    three backticks reach the gate one at a time, and a lone backtick already
    on the screen cannot be taken back once the other two turn it into a fence
    around a tool call.
    """
    if (fence := open_fence(text)) is not None:
        return fence
    # A strict prefix only. A complete ``` on the last line with the fences
    # already balanced is a closer, not an opener, and holding it would keep
    # back a block that has entirely arrived.
    start = text.rfind("\n") + 1
    tail = text[start:].strip()
    partial = tail and len(tail) < len(FENCE) and FENCE.startswith(tail)
    return start if partial else None


def visible(text: str, final: bool = True) -> str:
    """The part of ``text`` a reader should see.

    With ``final=False`` the scan stops at the first opener that has not closed
    yet, since more text could still turn it into a tool call. With
    ``final=True`` nothing is pending any more and an unclosed opener is prose,
    which is precisely :func:`parsing.strip_json` before its final trim.
    """
    # A fence whose contents have not arrived yet settles nothing after it.
    # With `final=True` nothing more is coming, so it was prose.
    if not final and (fence := _held(text)) is not None:
        text = text[:fence]

    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in OPENERS:
            end = scan_balanced(text, i)
            if end is None:
                if not final:
                    break
            else:
                try:
                    json.loads(text[i:end])
                except json.JSONDecodeError:
                    pass
                else:
                    i = end
                    continue
        out.append(text[i])
        i += 1
    return drop_empty_fences("".join(out))


class Gate:
    """Feeds streamed chunks out as prose, holding back anything JSON-shaped."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.seen = ""
        self.shown = 0
        self.wrote = False

    def feed(self, text: str) -> None:
        if not text:
            return
        self.seen += text
        self._flush(visible(self.seen, final=False))

    def finish(self) -> None:
        """Release anything still held, and reset for the next turn.

        One gate serves every turn of a session, so everything per-turn has to
        be cleared here -- including ``wrote``, or the second turn keeps the
        first one's opinion that prose has already started and stops trimming
        the blank space before a leading tool call.
        """
        self._flush(visible(self.seen, final=True))
        self.seen = ""
        self.shown = 0
        self.wrote = False

    def _flush(self, settled: str) -> None:
        # Leading whitespace before the first real character is swallowed, so a
        # reply that opens with a tool call does not begin with a blank line.
        pending = settled[self.shown :]
        if not self.wrote:
            stripped = pending.lstrip()
            self.shown += len(pending) - len(stripped)
            pending = stripped
        if not pending:
            return
        self.wrote = True
        self.shown += len(pending)
        self.emit(pending)
