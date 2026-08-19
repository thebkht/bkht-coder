"""What the terminal is, and how to talk to it.

Every other module asks this one whether there is a terminal, rather than
calling ``isatty`` itself. One answer, in one place: a session that is a
terminal for the spinner but not for the approval prompt would be a bug nobody
would think to look for.
"""

from __future__ import annotations

import shutil
import sys

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

CLEAR_LINE = "\r\033[2K"
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


def paint(text: str, colour: str, stream=None) -> str:
    """Colour ``text`` only when the destination is a terminal."""
    stream = stream or sys.stdout
    try:
        tty = stream.isatty()
    except (AttributeError, ValueError):
        tty = False
    return f"{colour}{text}{RESET}" if tty else text


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
