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

RULE = "\u2500"


def rule(width: int) -> str:
    """A horizontal run of ``width`` columns, for dividing a column's sections."""
    return RULE * max(0, width)


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
