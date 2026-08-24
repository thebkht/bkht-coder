"""The prompt line, drawn and edited by us rather than by readline.

``readline`` gives history and completion for free, and for a long time that
was enough. It cannot give two things this prompt now needs: a key that is not
a character -- Shift+Tab -- routed to a handler of ours, and a footer under the
input that changes while you type. Its macros can insert text on a keypress but
not submit it, and libedit (which is what ``import readline`` gets you on macOS)
will not even do that reliably.

So the terminal is put in cbreak mode and every keystroke is read here. That
buys the shortcut and the live footer, and costs us the editing keys, which are
implemented below -- the common half of readline, not all of it. Anywhere raw
mode is unavailable (a pipe, Windows, an IDE console) nothing here runs and
:class:`~bkht.coder.prompt.Reader` falls back to ``input()`` as before.

The block on screen is a rule, the input, a rule, and the footer. Rules rather
than a box: a full border has to pad every row to the same width, which stops
being simple the moment the text wraps.
"""

from __future__ import annotations

import codecs
import os
import sys
from collections.abc import Callable, Sequence

from . import banner, terminal
from .terminal import DIM

ESC = "\x1b"

#: What the terminal sends for the keys we answer to. Terminals disagree about
#: Shift+Tab -- xterm and everything descended from it send CSI Z, a few older
#: ones send ESC TAB -- so both are taken to mean the same thing.
BACK_TAB = ("[Z", "\t")


def available(stdin=None, stdout=None) -> bool:
    """True when this terminal can be driven a keystroke at a time."""
    if not terminal.interactive(stdout, stdin):
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except ImportError:
        return False
    stdin = sys.stdin if stdin is None else stdin
    try:
        stdin.fileno()
    except (AttributeError, ValueError, OSError):
        return False
    return True


class Editor:
    """One prompt line, read a keystroke at a time.

    ``footer`` is called on every redraw rather than passed in as text: what it
    says -- the permission mode -- is exactly what ``cycle`` changes, and a
    footer that had to be refreshed by its caller would be a footer that lies
    for one keypress.
    """

    def __init__(
        self,
        *,
        completions: Callable[[], Sequence[str]] | None = None,
        footer: Callable[[], str] | None = None,
        cycle: Callable[[], None] | None = None,
        history: list[str] | None = None,
        stdin=None,
        stdout=None,
    ) -> None:
        self.completions = completions or (lambda: ())
        self.footer = footer or (lambda: "")
        self.cycle = cycle
        self.history = history if history is not None else []
        self.stdin = sys.stdin if stdin is None else stdin
        self.stdout = sys.stdout if stdout is None else stdout
        self.buffer = ""
        self.cursor = 0
        self.drawn = 0  # rows the last redraw painted
        self.caret = 0  # which of those rows the cursor was left on
        self.recall = 0  # how far up the history the arrows have walked
        self.draft = ""  # what was being typed before they started

    # --- reading ------------------------------------------------------------

    def read(self, prompt: str = "› ") -> str:
        """Read one line. Ctrl-D on an empty line raises ``EOFError``."""
        import termios
        import tty

        self.buffer = ""
        self.cursor = 0
        self.drawn = 0
        self.caret = 0
        self.recall = len(self.history)
        self.draft = ""

        fd = self.stdin.fileno()
        saved = termios.tcgetattr(fd)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            # cbreak, not raw: Ctrl-C must stay a signal. An interrupt that
            # arrived as a byte would leave the loop below deciding what
            # Ctrl-C means, and it would decide it differently from the rest
            # of the session.
            tty.setcbreak(fd)
            self._redraw(prompt)
            while True:
                keys = decoder.decode(os.read(fd, 1024))
                if not keys:
                    continue
                line = self._consume(keys, prompt)
                if line is None:
                    self._redraw(prompt)
                    continue
                return line
        except KeyboardInterrupt:
            # Ctrl-C abandons the line, so the frame around it goes too --
            # left behind, it would sit in the scrollback still offering to
            # cycle a mode for a prompt that no longer exists.
            self._erase()
            self.stdout.write("\n")
            self.stdout.flush()
            raise
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _consume(self, keys: str, prompt: str) -> str | None:
        """Apply a chunk of input. Returns the finished line, or ``None``.

        A chunk rather than a key: a paste arrives as one read, and so does
        anything typed faster than the loop goes round.
        """
        index = 0
        while index < len(keys):
            key = keys[index]
            index += 1
            if key == ESC:
                sequence, index = self._escape(keys, index)
                self._control(sequence)
                continue
            if key in ("\r", "\n"):
                return self._submit(prompt)
            if key == "\t":
                self._complete()
            elif key == "\x7f" or key == "\b":
                self._backspace()
            elif key == "\x04":  # Ctrl-D
                if not self.buffer:
                    self._erase()
                    raise EOFError
            elif key == "\x01":  # Ctrl-A
                self.cursor = 0
            elif key == "\x05":  # Ctrl-E
                self.cursor = len(self.buffer)
            elif key == "\x0b":  # Ctrl-K
                self.buffer = self.buffer[: self.cursor]
            elif key == "\x15":  # Ctrl-U
                self.buffer = self.buffer[self.cursor :]
                self.cursor = 0
            elif key == "\x17":  # Ctrl-W
                self._kill_word()
            elif key >= " ":
                self._insert(key)
        return None

    def _escape(self, keys: str, index: int) -> tuple[str, int]:
        """The rest of an escape sequence, and where it ended.

        Terminated by the first byte that can end one -- a letter, or ``~`` --
        so an unknown sequence is swallowed whole rather than leaving its tail
        to be typed into the line.
        """
        if index >= len(keys):
            return "", index
        if keys[index] not in ("[", "O"):
            return keys[index], index + 1
        end = index + 1
        while end < len(keys) and not (keys[end].isalpha() or keys[end] == "~"):
            end += 1
        return keys[index : end + 1], min(end + 1, len(keys))

    def _control(self, sequence: str) -> None:
        if sequence in BACK_TAB:
            if self.cycle is not None:
                self.cycle()
        elif sequence == "[D":
            self.cursor = max(0, self.cursor - 1)
        elif sequence == "[C":
            self.cursor = min(len(self.buffer), self.cursor + 1)
        elif sequence == "[A":
            self._history(-1)
        elif sequence == "[B":
            self._history(1)
        elif sequence in ("[H", "OH", "[1~"):
            self.cursor = 0
        elif sequence in ("[F", "OF", "[4~"):
            self.cursor = len(self.buffer)
        elif sequence == "[3~":
            self.buffer = self.buffer[: self.cursor] + self.buffer[self.cursor + 1 :]

    # --- editing ------------------------------------------------------------

    def _insert(self, text: str) -> None:
        self.buffer = self.buffer[: self.cursor] + text + self.buffer[self.cursor :]
        self.cursor += len(text)

    def _backspace(self) -> None:
        if self.cursor:
            self.buffer = self.buffer[: self.cursor - 1] + self.buffer[self.cursor :]
            self.cursor -= 1

    def _kill_word(self) -> None:
        head = self.buffer[: self.cursor].rstrip()
        head = head[: head.rfind(" ") + 1] if " " in head else ""
        self.buffer = head + self.buffer[self.cursor :]
        self.cursor = len(head)

    def _history(self, step: int) -> None:
        """Walk the history, keeping whatever was typed before the first step.

        The draft is held rather than overwritten so that pressing Up to check
        the last prompt is not a way to lose the one being written.
        """
        if not self.history:
            return
        if self.recall == len(self.history):
            self.draft = self.buffer
        position = max(0, min(len(self.history), self.recall + step))
        self.recall = position
        self.buffer = self.draft if position == len(self.history) else self.history[position]
        self.cursor = len(self.buffer)

    def _complete(self) -> None:
        """Complete a slash command, and nothing else.

        The same rule readline was given: completion off a bare word would
        offer filenames, which is not what a line here usually is.
        """
        head = self.buffer[: self.cursor]
        if not head.startswith("/") or " " in head:
            return
        matches = [option for option in self.completions() if option.startswith(head)]
        if not matches:
            return
        shared = os.path.commonprefix(matches)
        if len(shared) > len(head):
            self._insert(shared[len(head) :])
        elif len(matches) > 1:
            self._above("  ".join(matches))

    # --- drawing ------------------------------------------------------------

    def _rows(self, prompt: str, width: int) -> list[str]:
        """The block as it should appear, top rule to footer."""
        rule = terminal.paint(banner.rule(width), DIM, self.stdout)
        text = f"{terminal.paint(prompt, terminal.ACCENT, self.stdout)}{self.buffer}"
        footer = self.footer()
        rows = [rule, text, rule]
        if footer:
            rows.append(footer)
        return rows

    def _wrapped(self, prompt: str, width: int) -> int:
        """How many screen rows the input line takes up."""
        columns = terminal.visible(prompt) + len(self.buffer)
        return max(1, -(-columns // width)) if width else 1

    def _redraw(self, prompt: str) -> None:
        width = max(20, terminal.width())
        self._home()
        rows = self._rows(prompt, width)
        text_rows = self._wrapped(prompt, width)
        self.stdout.write("\n".join(rows))

        # Where the caret belongs, counted from the top of the block: past the
        # rule, then wherever the cursor sits in the (possibly wrapped) input.
        offset = terminal.visible(prompt) + self.cursor
        caret_row = 1 + offset // width
        total = 1 + text_rows + 1 + (len(rows) - 3)
        up = total - 1 - caret_row
        if up > 0:
            self.stdout.write(f"{ESC}[{up}A")
        self.stdout.write("\r")
        if column := offset % width:
            self.stdout.write(f"{ESC}[{column}C")
        self.stdout.flush()
        self.drawn, self.caret = total, caret_row

    def _home(self) -> None:
        """Put the cursor at the top-left of the block and clear what is there."""
        if self.caret:
            self.stdout.write(f"{ESC}[{self.caret}A")
        self.stdout.write(f"\r{ESC}[J")

    def _erase(self) -> None:
        self._home()
        self.stdout.flush()
        self.drawn = self.caret = 0

    def _above(self, text: str) -> None:
        """Print something in the scrollback, above the block.

        Completion candidates and nothing else so far. They go above rather
        than below because below is where the footer is, and a list that pushed
        the footer around would make the mode look like it was moving.
        """
        self._erase()
        self.stdout.write(f"{terminal.paint(text, DIM, self.stdout)}\n")

    def _submit(self, prompt: str) -> str:
        """Leave the finished line in the scrollback and hand it back.

        The rules and the footer are chrome for the moment you are typing; kept
        in the transcript they would say things about a line that has already
        been answered.
        """
        line = self.buffer
        self.buffer, self.cursor, self.draft = "", 0, ""
        self._erase()
        rendered = terminal.paint(prompt, terminal.ACCENT, self.stdout)
        self.stdout.write(f"{rendered}{line}\n")
        self.stdout.flush()
        if line.strip():
            self.history.append(line)
        self.recall = len(self.history)
        return line


def footer(mode: str, cycles: bool = True, stream=None) -> str:
    """The line under the prompt: what mode this is, and how to change it."""
    text = f"{mode} mode on"
    if cycles:
        text += " (shift+tab to cycle)"
    return f"{terminal.paint('▸▸', terminal.ORANGE, stream)} {terminal.paint(text, DIM, stream)}"
