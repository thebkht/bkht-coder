"""What the terminal is, and how to talk to it.

Every other module asks this one whether there is a terminal, rather than
calling ``isatty`` itself. One answer, in one place: a session that is a
terminal for the spinner but not for the approval prompt would be a bug nobody
would think to look for.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
REVERSE = "\033[7m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"


def supports_256(env=None) -> bool:
    """True when the terminal can be trusted with the 256-colour range.

    The two accents this file exposes are hues the sixteen ANSI colours cannot
    reach, and a terminal that does not understand ``38;5`` renders the escape
    as text. Asked once, at import, because the answer cannot change while the
    process runs.
    """
    env = os.environ if env is None else env
    if (env.get("COLORTERM") or "").lower() in ("truecolor", "24bit"):
        return True
    return "256color" in (env.get("TERM") or "")


def _hue(extended: str, basic: str) -> str:
    return extended if supports_256() else basic


#: The two accents the interface is built from: orange names what the agent is
#: doing, blue names what the user asked and how the answer is structured.
#: Everything else on screen is weight, or dim.
ORANGE = _hue("\033[38;5;208m", YELLOW)
ACCENT = _hue("\033[38;5;33m", BLUE)

CLEAR_LINE = "\r\033[2K"
CURSOR_UP = "\033[A"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def interactive(stdout=None, stdin=None) -> bool:
    """True when both ends are a terminal, so live rendering is safe.

    Both, not either: output redirected to a file must not grow spinner frames,
    and input from a pipe cannot answer a single-key prompt.
    """
    stdout = sys.stdout if stdout is None else stdout
    stdin = sys.stdin if stdin is None else stdin
    try:
        return bool(stdout.isatty() and stdin.isatty())
    except (AttributeError, ValueError):
        # ValueError: the stream was closed underneath us.
        return False


def colourful(stream=None) -> bool:
    """True when ``stream`` should be given escape codes at all.

    Asked directly by anything that colours more than one span and has to know
    up front -- a box paints its border but not the text a caller handed it,
    and the two must agree about whether this is a terminal.
    """
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def paint(text: str, colour: str, stream=None) -> str:
    """Colour ``text`` only when the destination is a terminal."""
    return f"{colour}{text}{RESET}" if colourful(stream) else text


#: Every SGR sequence ``paint`` can emit. Colour is invisible to the terminal's
#: column count, and to anything padding a column by hand.
SGR = re.compile(r"\033\[[0-9;]*m")


def visible(text: str) -> int:
    """How many columns ``text`` takes up once the colour is stripped.

    ``len`` is the wrong answer for anything ``paint`` has touched: a coloured
    word carries a dozen bytes the terminal never draws, and padding by ``len``
    puts every column after it out by that much.
    """
    return len(SGR.sub("", text))


def fit(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` columns, with an ellipsis where it was cut.

    Cutting, not wrapping: a line that runs past its column in a boxed layout
    has nowhere to wrap to, and a second row would push the border out of
    square. Text with colour in it is returned whole rather than cut mid-escape
    -- the callers that truncate are the ones building their own columns, and
    they colour the result afterwards.
    """
    if limit <= 0:
        return ""
    if visible(text) <= limit or SGR.search(text):
        return text
    return text[: limit - 1].rstrip() + "…" if limit > 1 else "…"


#: What a full and an empty cell of a meter are drawn with. Half-blocks rather
#: than a bracketed run of hashes: the meter sits inside a line of prose and has
#: to read as a bar at a glance without a border to say where it starts.
#:
#: The pair after them is what a console that cannot encode blocks gets. A
#: Windows terminal left on cp1252 raises on the first one rather than drawing
#: it badly, and the prompt is not the place to discover that.
FULL, EMPTY = "\u2588", "\u2591"
ASCII_FULL, ASCII_EMPTY = "#", "-"


def encodable(text: str, stream=None) -> bool:
    """True when ``stream`` can actually carry ``text``.

    Asked of anything drawn outside Latin-1, because the failure is not a
    fallback glyph -- it is a ``UnicodeEncodeError`` out of the write itself, or
    mojibake where a meter should be.
    """
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def bar(ratio: float, cells: int = 12, stream=None) -> str:
    """A meter ``cells`` wide, ``ratio`` of it filled.

    Uncoloured, because who colours it is the caller: what a full bar *means*
    differs by what is being measured, and this only knows how many cells it is.

    A ratio above one fills the bar rather than overflowing it, and one just
    above zero still lights a cell -- a meter that reads empty while something
    is being consumed is worse than one that rounds up.
    """
    cells = max(0, cells)
    ratio = min(1.0, max(0.0, ratio))
    filled = round(ratio * cells)
    if ratio > 0 and filled == 0:
        filled = 1
    full, empty = (FULL, EMPTY) if encodable(FULL + EMPTY, stream) else (ASCII_FULL, ASCII_EMPTY)
    return full * filled + empty * (cells - filled)


def width(default: int = 80) -> int:
    """Terminal columns, for truncating anything drawn on a single line."""
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        return default


def read_key(stdin=None) -> str:
    """Read one keypress without waiting for Enter.

    Falls back to a whole line wherever raw mode is unavailable -- a pipe, a
    Windows box without ``msvcrt``, an IDE console -- so callers never need a
    second path for "this is not really a terminal".
    """
    stdin = sys.stdin if stdin is None else stdin

    if not interactive(sys.stdout, stdin):
        return _read_line(stdin)

    try:
        import termios
        import tty
    except ImportError:
        return _read_key_windows(stdin)

    try:
        fd = stdin.fileno()
        saved = termios.tcgetattr(fd)
    except Exception:
        return _read_line(stdin)

    try:
        # cbreak, not raw: Ctrl-C must keep interrupting rather than arriving
        # as a byte nobody handles.
        tty.setcbreak(fd)
        return stdin.read(1)
    except Exception:
        return ""
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:
            pass


def _read_key_windows(stdin) -> str:
    try:
        import msvcrt
    except ImportError:
        return _read_line(stdin)
    try:
        return msvcrt.getwch()
    except Exception:
        return _read_line(stdin)


def _read_line(stdin) -> str:
    try:
        line = stdin.readline()
    except (EOFError, KeyboardInterrupt, ValueError):
        return ""
    return line.strip()[:1] if line.strip() else ""
