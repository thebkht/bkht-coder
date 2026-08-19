"""The approval prompt, answered with one key.

Approving a change is the most frequent thing anyone does in an `ask`-mode
session, and `y<Enter>` for every call is enough friction that the honest
response is to give up and run `--auto` -- which is the one outcome the
permission system exists to avoid.

`d` is here for the same reason: a preview cut off at forty lines invites
approving the part you cannot see.
"""

from __future__ import annotations

import contextlib
import sys

from .highlight import diff
from .permissions import truncate
from .terminal import CYAN, DIM, RESET, read_key

HINT = "[y] yes  [n] no  [a] always  [d] full diff"

YES = {"y", "Y"}
NO = {"n", "N"}
ALWAYS = {"a", "A"}
DIFF = {"d", "D"}


def ask_tty(question: str, body: str, keys=read_key, out=None, colour: bool | None = None,
            pause=None) -> str:
    """Show the change, then take a single keypress. Returns y / n / a.

    Anything unrecognised -- Enter, Ctrl-C, a closed stream -- is a refusal,
    matching the [y/N] default that the typed prompt has always had. Silence
    must never mean yes.

    ``pause`` suspends the status line for the whole exchange. Without it the
    spinner keeps repainting over the diff being approved, which is the one
    piece of output that has to survive long enough to be read.
    """
    stream = out or sys.stdout
    write = _writer(stream)
    if colour is None:
        colour = getattr(stream, "isatty", bool)()
    with (pause or contextlib.nullcontext)():
        return _ask(write, question, body, keys, colour)


def _ask(write, question, body, keys, colour) -> str:
    expanded = False
    while True:
        if body:
            write(diff(body if expanded else truncate(body), colour=colour))
        write(_paint(f"{question} ", CYAN, colour) + _paint(HINT, DIM, colour))

        key = keys()
        if key in DIFF and body and not expanded:
            expanded = True
            continue

        answer = "y" if key in YES else "a" if key in ALWAYS else "n"
        # Echoed because the keypress itself leaves no mark: scrollback should
        # still show what was approved and what was not.
        write(_paint(f"  {_label(answer)}", DIM, colour))
        return answer


def _paint(text: str, colour_code: str, colour: bool) -> str:
    """Colour by the stream actually being written to, not by sys.stdout."""
    return f"{colour_code}{text}{RESET}" if colour else text


def _label(answer: str) -> str:
    return {"y": "yes", "a": "always", "n": "no"}[answer]


def _writer(stream):
    def write(text: str) -> None:
        print(text, file=stream, flush=True)

    return write
