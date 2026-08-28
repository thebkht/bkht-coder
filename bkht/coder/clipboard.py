"""Getting an image off the system clipboard.

Terminals do not deliver images. A paste is text, and on every platform the
picture on the clipboard reaches a program through the platform's own API --
so this shells out to whatever the box already has, the way `terminal.py`
answers questions about the terminal so nothing else has to ask.

Nothing here is a dependency. Pillow would read the clipboard on Linux and not
on macOS; `pbpaste` cannot do images at all. What works is a small command per
platform, and a clear answer when none of them is installed -- because the
alternative to saying so is a keypress that silently does nothing.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .session import STATE_DIR

#: Where a pasted image is kept. Under the state directory rather than in a
#: temp dir: the path goes into the transcript, and a transcript that points at
#: files the next boot has deleted is a transcript that cannot be resumed.
IMAGE_DIR = STATE_DIR / "images"

#: Long enough for a screenshot, short enough that a hung helper does not look
#: like a hung session.
TIMEOUT = 10.0

#: macOS. AppleScript is the only route to the clipboard's image data without a
#: compiled helper; it hands back `«data PNGf89504e47...»`, hex inside guards.
MACOS = [
    "osascript", "-e",
    'try\nset the clipboard to (the clipboard as «class PNGf»)\nend try\n'
    'get the clipboard as «class PNGf»',
]

#: Wayland, then X11. Both write the bytes straight to stdout.
WAYLAND = ["wl-paste", "--type", "image/png", "--no-newline"]
X11 = ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]


def _run(argv: list[str], text: bool = False) -> bytes | str | None:
    if not shutil.which(argv[0]):
        return None
    try:
        done = subprocess.run(
            argv, capture_output=True, timeout=TIMEOUT,
            **({"encoding": "utf-8", "errors": "replace"} if text else {}),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not done.stdout:
        return None
    return done.stdout


def _from_applescript(raw: str) -> bytes | None:
    """The PNG inside `«data PNGf89504e470d0a1a0a...»`."""
    start = raw.find("PNGf")
    if start == -1:
        return None
    end = raw.find("»", start)
    hexed = raw[start + 4 : end if end != -1 else len(raw)].strip()
    try:
        return bytes.fromhex(hexed)
    except ValueError:
        return None


def read_image() -> bytes | None:
    """The PNG on the clipboard, or ``None`` when there is not one.

    ``None`` covers both "nothing copied" and "no helper installed"; the caller
    tells them apart with :func:`helper_missing`, because only one of them is
    worth a sentence about what to install.
    """
    if sys.platform == "darwin":
        raw = _run(MACOS, text=True)
        return _from_applescript(raw) if raw else None
    for argv in (WAYLAND, X11):
        if (data := _run(argv)) is not None:
            return data
    return None


def helper_missing() -> str:
    """What to install to make image paste work here, or ``""`` if it can.

    Windows is not answered: the editor that would read the keystroke needs
    termios, so there is no path on which this could be asked there.
    """
    if sys.platform == "darwin":
        return "" if shutil.which("osascript") else "osascript (part of macOS)"
    if sys.platform.startswith("linux"):
        if shutil.which("wl-paste") or shutil.which("xclip"):
            return ""
        return "wl-clipboard (Wayland) or xclip (X11)"
    return "no clipboard helper is known for this platform"


def looks_like_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def save(data: bytes, directory: Path | None = None) -> Path:
    """Write ``data`` where it can be pointed at, and return the path."""
    directory = IMAGE_DIR if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{len(data) % 100000:05d}.png"
    path.write_bytes(data)
    return path


def encode(path: Path | str) -> str:
    """A saved image as base64, which is how Ollama takes one."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")
