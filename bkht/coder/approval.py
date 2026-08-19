"""The approval prompt, answered with one key.

Approving a change is the most frequent thing anyone does in an `ask`-mode
session, and `y<Enter>` for every call is enough friction that the honest
response is to give up and run `--auto` -- which is the one outcome the
permission system exists to avoid.

`d` is here for the same reason: a preview cut off at forty lines invites
approving the part you cannot see.
"""

from __future__ import annotations

import sys

from .highlight import diff
from .permissions import truncate
from .terminal import CYAN, DIM, paint, read_key

HINT = "[y] yes  [n] no  [a] always  [d] full diff"

YES = {"y", "Y"}
NO = {"n", "N"}
ALWAYS = {"a", "A"}
DIFF = {"d", "D"}


def ask_tty(question: str, body: str, keys=read_key, out=None, colour: bool | None = None) -> str:
    """Show the change, then take a single keypress. Returns y / n / a.

    Anything unrecognised -- Enter, Ctrl-C, a closed stream -- is a refusal,
    matching the [y/N] default that the typed prompt has always had. Silence
    must never mean yes.
    """
    stream = out or sys.stdout
    write = _writer(stream)
    if colour is None:
        colour = getattr(stream, "isatty", bool)()
    expanded = False

    while True:
        if body:
            write(diff(body if expanded else truncate(body), colour=colour))
        write(paint(f"{question} ", CYAN) + paint(HINT, DIM))

        key = keys()
        if key in DIFF and body and not expanded:
            expanded = True
            continue

        answer = "y" if key in YES else "a" if key in ALWAYS else "n"
        # Echoed because the keypress itself leaves no mark: scrollback should
        # still show what was approved and what was not.
        write(paint(f"  {_label(answer)}", DIM))
        return answer


def _label(answer: str) -> str:
    return {"y": "yes", "a": "always", "n": "no"}[answer]


def _writer(stream):
    def write(text: str) -> None:
        print(text, file=stream, flush=True)

    return write
