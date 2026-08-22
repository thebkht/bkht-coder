"""The mark, drawn once, in braille.

The logo is a 16x16 square frame with two opposite corners cut away. A braille
cell holds 2 dots across and 4 down, so at one dot per unit the whole mark fits
in eight columns and four rows and its edges land mid-cell, which is what gives
the partial glyphs their shape.

Eight columns rather than sixteen: the greeting is chrome above the first
prompt, and chrome that takes a third of the screen is chrome that gets in the
way of the thing it introduces.

It lives here as a literal rather than as geometry rasterised at startup: the
art never changes, and a banner is not worth a loop that runs before the first
prompt. Both the README header and the interactive greeting are this file.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import terminal

BLANK = "⠀"

LOGO: tuple[str, ...] = (
    "⣀⣸⣿⣿⣿⣿⣿⣿",
    "⣿⣿⠀⠀⠀⠀⣿⣿",
    "⣿⣿⠀⠀⠀⠀⣿⣿",
    "⣿⣿⣿⣿⣿⣿⡏⠉",
)

WIDTH = 8
GUTTER = 5

#: Art, gutter, and enough room for the longest line printed beside it.
MIN_WIDTH = 54

#: Under this the two columns would each be too narrow to say anything, so the
#: caller draws the art with the text beside it instead. The box is the widest
#: of the three greetings, not the only one.
BOX_MIN_WIDTH = 80

#: A greeting that grows to fill a maximised terminal is a greeting whose two
#: columns drift a screen apart. Past this the box stops widening.
MAX_WIDTH = 100

#: Blank columns between a border and the text inside it.
PAD = 2

TOP_LEFT = "\u256d"
TOP_RIGHT = "\u256e"
BOTTOM_LEFT = "\u2570"
BOTTOM_RIGHT = "\u256f"
EDGE = "\u2502"
RULE = "\u2500"


def rule(width: int) -> str:
    """A horizontal run of ``width`` columns, for dividing a column's sections."""
    return RULE * max(0, width)


def _cell(text: str | None, width: int, centre: bool = False) -> str:
    """``text`` in a column exactly ``width`` columns wide.

    Measured with ``terminal.visible`` rather than ``len``, so a line that has
    already been coloured pads to the same place an uncoloured one would.
    """
    text = terminal.fit(text or "", width)
    gap = max(0, width - terminal.visible(text))
    if not centre:
        return text + " " * gap
    left = gap // 2
    return " " * left + text + " " * (gap - left)


def frame(
    title: str,
    left: Sequence[str | None],
    right: Sequence[str | None],
    width: int,
    colour: bool = True,
) -> str:
    """Two columns of text in a box ``width`` columns wide, ``title`` in its lid.

    ``left`` is centred and ``right`` is ragged-right, which is what makes the
    shape read as a mark with a page beside it rather than as a table. ``None``
    is a blank row, the same convention ``render`` uses, and the shorter column
    is padded with blank rows so both reach the bottom.

    The left column is sized to its own widest line -- never narrower than the
    art -- and the right column takes what is left. Anything still too long is
    cut by ``terminal.fit``: a box whose contents wrap is a box with a hole in
    one side.

    ``colour`` off produces the same geometry with no escape bytes in it, so
    the shape can be measured without stripping them first.
    """
    def tint(text: str, code: str) -> str:
        return f"{code}{text}{terminal.RESET}" if colour else text

    interior = width - 2
    widest = max([WIDTH, *(terminal.visible(line) for line in left if line)])
    # Half the box at the outside: the left column holds a mark and three short
    # facts, and a column wider than that is one long path away from crowding
    # the text it was meant to introduce.
    cell_left = min(widest + PAD * 2, interior // 2)
    cell_right = interior - cell_left - 1

    rows = []
    for index in range(max(len(left), len(right))):
        here = left[index] if index < len(left) else None
        there = right[index] if index < len(right) else None
        rows.append(
            tint(EDGE, terminal.ORANGE)
            + _cell(here, cell_left, centre=True)
            + tint(EDGE, terminal.ORANGE)
            + " " * PAD
            + _cell(there, cell_right - PAD * 2)
            + " " * PAD
            + tint(EDGE, terminal.ORANGE)
        )

    # The lid carries the title so the box needs no first row to say what it
    # is; the run of rule after it fills whatever the title did not.
    lead = f"{TOP_LEFT}{RULE} {title} " if title else TOP_LEFT
    lid = tint(lead + rule(width - terminal.visible(lead) - 1) + TOP_RIGHT, terminal.ORANGE)
    floor = tint(BOTTOM_LEFT + rule(interior) + BOTTOM_RIGHT, terminal.ORANGE)
    return "\n".join([lid, *rows, floor])


def render(lines: Sequence[str | None]) -> str:
    """The logo with ``lines`` set beside it, one entry per row.

    ``None`` -- or running out of entries -- leaves that row bare, and a bare
    row keeps no trailing whitespace, so the block can be pasted anywhere
    without a ragged right edge.

    More entries than the art has rows is not an error: the extra ones carry on
    below it, indented to the same left edge. The art is four rows now, and a
    greeting with more to say than that should not have to split itself across
    two calls to line up.
    """
    out = []
    for index in range(max(len(LOGO), len(lines))):
        art = LOGO[index].ljust(WIDTH, BLANK) if index < len(LOGO) else " " * WIDTH
        text = lines[index] if index < len(lines) else None
        if text:
            out.append(f"{art}{' ' * GUTTER}{text}")
        else:
            out.append(art.rstrip())
    return "\n".join(out)


def drawable(stream=None) -> bool:
    """True when ``stream`` can carry braille.

    A terminal in a non-UTF-8 locale would take the banner as mojibake, or take
    the exception instead -- either way the greeting is not the place to find
    out.
    """
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        LOGO[0].encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True
