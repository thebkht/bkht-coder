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

The input is one buffer that may contain newlines, not one line. That is what
makes a pasted diff survivable: without bracketed paste a forty-line paste
submits its first line and types the other thirty-nine into the prompts that
follow, each of them a task the agent then tries to do.
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

#: Bracketed paste. The terminal wraps pasted text in these, which is the only
#: way to tell a newline that was pasted from one that was typed -- and the
#: difference between them is the difference between one prompt and forty.
PASTE_ON = "\x1b[?2004h"
PASTE_OFF = "\x1b[?2004l"
PASTE_START = "[200~"
PASTE_END = "[201~"

#: What continuation lines sit behind, so the text stays aligned under the
#: first line rather than under the prompt mark.
CONTINUATION = "  "

#: What an attached image looks like in the line. A chip, like a long paste,
#: because the thing itself is not text and has no business being in a buffer.
IMAGE_CHIP = "[Image #{number}]"

#: A paste longer than this is folded. Redrawing five hundred lines inside the
#: block on every keystroke is not editing, it is flicker; and the point of
#: pasting a file is rarely to edit it afterwards.
PASTE_LINES = 4

#: What is left in place of a paste a fold takes out. One line, numbered, with
#: the count that says how much it stands for: the block is there to be talked
#: about, not read, and a chip that stays one row keeps it that way.
PASTE_CHIP = "[Pasted text #{number}, {lines} lines]"


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
        attach: Callable[[], str | None] | None = None,
        on_image: Callable[[str], None] | None = None,
        stdin=None,
        stdout=None,
    ) -> None:
        self.completions = completions or (lambda: ())
        self.footer = footer or (lambda: "")
        self.cycle = cycle
        #: Returns the path of an image taken off the clipboard, or None.
        #: Injected so the editor never has to know what a clipboard is.
        self.attach = attach
        self.history = history if history is not None else []
        self.stdin = sys.stdin if stdin is None else stdin
        self.stdout = sys.stdout if stdout is None else stdout
        self.buffer = ""
        self.cursor = 0
        self.drawn = 0  # rows the last redraw painted
        self.caret = 0  # which of those rows the cursor was left on
        self.recall = 0  # how far up the history the arrows have walked
        self.draft = ""  # what was being typed before they started
        self.pending = ""  # an escape sequence cut in half by the end of a read
        self.pasting = False  # between the bracketed-paste markers
        self.paste_at = 0  # where the current paste began in the buffer
        self.pastes: dict[int, tuple[str, str]] = {}  # number -> (chip, full text)
        self.images: list[str] = []  # paths pasted into the line being written
        #: Called with a path once an image is attached, so the session can say
        #: whether the model will actually look at it. Here rather than in the
        #: caller because the answer belongs beside the keypress that earned it.
        self.on_image = on_image or (lambda path: None)

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
        self.pending = ""
        self.pasting = False
        self.paste_at = 0
        self.pastes = {}
        self.images = []

        fd = self.stdin.fileno()
        saved = termios.tcgetattr(fd)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            # cbreak, not raw: Ctrl-C must stay a signal. An interrupt that
            # arrived as a byte would leave the loop below deciding what
            # Ctrl-C means, and it would decide it differently from the rest
            # of the session.
            tty.setcbreak(fd)
            # Asked for and given back inside the same block that owns cbreak:
            # a terminal left in bracketed paste after we stop reading would
            # spill `[200~` into whatever runs next.
            self.stdout.write(PASTE_ON)
            self.stdout.flush()
            self._redraw(prompt)
            while True:
                keys = decoder.decode(os.read(fd, 1024))
                if not keys:
                    continue
                line = self._consume(keys, prompt)
                if line is None:
                    self._paint(prompt)
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
            self.stdout.write(PASTE_OFF)
            self.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _consume(self, keys: str, prompt: str) -> str | None:
        """Apply a chunk of input. Returns the finished line, or ``None``.

        A chunk rather than a key: a paste arrives as one read, and so does
        anything typed faster than the loop goes round.
        """
        keys, self.pending = self.pending + keys, ""
        index = 0
        while index < len(keys):
            key = keys[index]
            index += 1
            if key == ESC:
                sequence, index = self._escape(keys, index)
                if sequence is None:
                    # The read ended mid-sequence. Held rather than acted on:
                    # half of `[201~` taken for text is a paste that never
                    # ends and a block that never folds.
                    self.pending = keys[index:]
                    break
                self._control(sequence)
                continue
            if key in ("\r", "\n"):
                # A newline that was pasted is text; one that was typed is the
                # decision to send. Only the terminal knows which this is, and
                # only while bracketed paste is on.
                if self.pasting:
                    self._insert("\n")
                    continue
                return self._submit(prompt)
            if key == "\t":
                if self.pasting:
                    self._insert("\t")
                else:
                    self._complete()
            elif key == "\x7f" or key == "\b":
                self._backspace()
            elif key == "\x04":  # Ctrl-D
                if not self.buffer:
                    self._erase()
                    raise EOFError
            elif key == "\x01":  # Ctrl-A
                self.cursor = self._line_bounds()[0]
            elif key == "\x05":  # Ctrl-E
                self.cursor = self._line_bounds()[1]
            elif key == "\x0b":  # Ctrl-K
                end = self._line_bounds()[1]
                self.buffer = self.buffer[: self.cursor] + self.buffer[end:]
            elif key == "\x15":  # Ctrl-U
                start = self._line_bounds()[0]
                self.buffer = self.buffer[:start] + self.buffer[self.cursor :]
                self.cursor = start
            elif key == "\x17":  # Ctrl-W
                self._kill_word()
            elif key == "\x16":  # Ctrl-V
                self._attach()
            elif key >= " ":
                self._insert(key)
        return None

    def _escape(self, keys: str, index: int) -> tuple[str | None, int]:
        """The rest of an escape sequence, and where it ended.

        Terminated by the first byte that can end one -- a letter, or ``~`` --
        so an unknown sequence is swallowed whole rather than leaving its tail
        to be typed into the line. ``None`` when a sequence that had started ran
        out with the chunk: it is not wrong, it is not all here yet, and the
        caller holds what there is until the rest arrives. A lone ESC at the
        end is not held -- that is the Escape key, and waiting on it would eat
        the next thing typed.
        """
        if index >= len(keys):
            return "", index
        if keys[index] not in ("[", "O"):
            return keys[index], index + 1
        end = index + 1
        while end < len(keys) and not (keys[end].isalpha() or keys[end] == "~"):
            end += 1
        if end >= len(keys):
            return None, index - 1
        return keys[index : end + 1], end + 1

    def _control(self, sequence: str) -> None:
        if sequence == PASTE_START:
            self.pasting = True
            self.paste_at = self.cursor
        elif sequence == PASTE_END:
            self.pasting = False
            self._fold_paste()
        elif sequence == "\r":
            # Alt/Option+Enter arrives as ESC CR. `_escape` already hands it
            # back whole; it just had nothing to land on before now.
            self._insert("\n")
        elif sequence in BACK_TAB:
            if self.cycle is not None:
                self.cycle()
        elif sequence == "[D":
            self.cursor = max(0, self.cursor - 1)
        elif sequence == "[C":
            self.cursor = min(len(self.buffer), self.cursor + 1)
        elif sequence == "[A":
            if not self._vertical(-1):
                self._history(-1)
        elif sequence == "[B":
            if not self._vertical(1):
                self._history(1)
        elif sequence in ("[H", "OH", "[1~"):
            self.cursor = self._line_bounds()[0]
        elif sequence in ("[F", "OF", "[4~"):
            self.cursor = self._line_bounds()[1]
        elif sequence == "[3~":
            self.buffer = self.buffer[: self.cursor] + self.buffer[self.cursor + 1 :]

    # --- lines ---------------------------------------------------------------
    #
    # The buffer is one string that may contain newlines, so "the line" is a
    # slice of it rather than a thing in its own right. Kept as functions of
    # the cursor rather than as state: a second copy of where the lines are is
    # a second thing that can be wrong.

    def _line_bounds(self, position: int | None = None) -> tuple[int, int]:
        """Where the line holding ``position`` starts and ends."""
        position = self.cursor if position is None else position
        start = self.buffer.rfind("\n", 0, position) + 1
        end = self.buffer.find("\n", position)
        return start, len(self.buffer) if end == -1 else end

    def _locate(self) -> tuple[int, int]:
        """The cursor as ``(line index, column within that line)``."""
        head = self.buffer[: self.cursor]
        return head.count("\n"), self.cursor - (head.rfind("\n") + 1)

    def _vertical(self, step: int) -> bool:
        """Move a line up or down. False when there is no line to move to.

        False rather than clamping, because the caller's fallback is history
        recall: Up on the top line has to keep meaning "the previous prompt",
        or a multi-line buffer would quietly cost you the history.
        """
        if "\n" not in self.buffer:
            return False
        start, end = self._line_bounds()
        if step < 0:
            if start == 0:
                return False
            target = self._line_bounds(start - 1)
        else:
            if end == len(self.buffer):
                return False
            target = self._line_bounds(end + 1)
        column = self.cursor - start
        self.cursor = min(target[0] + column, target[1])
        return True

    def _attach(self) -> None:
        """Take an image off the clipboard and put a chip in its place.

        Ctrl-V and not the terminal's own paste: a terminal pasting an image
        sends nothing at all, because there is no way to send one. So the
        clipboard has to be asked directly, and it needs a key of its own.
        """
        if self.attach is None:
            return
        path = self.attach()
        if not path:
            return
        self.images.append(path)
        self._insert(IMAGE_CHIP.format(number=len(self.images)))
        self.on_image(path)

    def _chip(self, number: int, text: str) -> str:
        """A long paste as the one line that stands for it on screen.

        Numbered, so a prompt carrying two of them can say which is which, and
        counted, so the line says how much it stands for. The text itself is
        put back on the way out; on screen it is a single row.
        """
        return PASTE_CHIP.format(number=number, lines=text.count("\n") + 1)

    def _fold_paste(self) -> None:
        """Fold a long paste down to the lines that stand for it.

        The text is kept, not lost -- :meth:`_submit` puts it back. What is
        avoided is holding five hundred lines in a block that is redrawn on
        every keypress, and asking someone to find their cursor in it.
        """
        text = self.buffer[self.paste_at : self.cursor]
        if text.count("\n") + 1 <= PASTE_LINES:
            return
        number = len(self.pastes) + 1
        chip = self._chip(number, text)
        self.pastes[number] = (chip, text)
        self.buffer = self.buffer[: self.paste_at] + chip + self.buffer[self.cursor :]
        self.cursor = self.paste_at + len(chip)

    def _unfold(self, line: str) -> str:
        """A submitted line with every folded paste put back in full."""
        for chip, text in self.pastes.values():
            line = line.replace(chip, text)
        return line

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

    def _prefixes(self, prompt: str) -> list[str]:
        """What sits in front of each line of the buffer.

        The prompt mark leads the first line; the rest are indented to line up
        under it, so a pasted block reads as one thing rather than as several
        prompts stacked.
        """
        lines = self.buffer.split("\n")
        painted = terminal.paint(prompt, terminal.ACCENT, self.stdout)
        return [painted] + [CONTINUATION] * (len(lines) - 1)

    def _rows(self, prompt: str, width: int) -> list[str]:
        """The block as it should appear, top rule to footer."""
        rule = terminal.paint(banner.rule(width), DIM, self.stdout)
        lines = self.buffer.split("\n")
        prefixes = self._prefixes(prompt)
        footer = self.footer()
        rows = [rule, *(prefix + line for prefix, line in zip(prefixes, lines)), rule]
        if footer:
            # Split rather than appended whole: the footer is one string so that
            # the callable stays one shape, but every row below has to be a row
            # here or the redraw walks back up the wrong number of them.
            rows.extend(footer.split("\n"))
        return rows

    def _wrapped(self, prompt: str, width: int) -> list[int]:
        """How many screen rows each line of the buffer takes up.

        A list rather than a total: the caret needs to know how many rows sit
        above its own line, which a total cannot say.
        """
        if not width:
            return [1] * (self.buffer.count("\n") + 1)
        counts = []
        for prefix, line in zip(self._prefixes(prompt), self.buffer.split("\n")):
            columns = terminal.visible(prefix) + len(line)
            counts.append(max(1, -(-columns // width)))
        return counts

    def _paint(self, prompt: str) -> None:
        """Redraw, unless a paste is still arriving.

        A long paste comes in over several reads, and painting each one draws
        a taller block than the last. Once the block is taller than the window
        the terminal scrolls, the cursor-up erase can no longer reach the top
        of what was drawn, and every frame is left behind -- one paste ends up
        looking like six. Nothing is drawn until the paste ends and folds.
        """
        if self.pasting:
            return
        self._redraw(prompt)

    def _redraw(self, prompt: str) -> None:
        width = max(20, terminal.width())
        self._home()
        rows = self._rows(prompt, width)
        counts = self._wrapped(prompt, width)
        self.stdout.write("\n".join(rows))

        # Where the caret belongs, counted from the top of the block: past the
        # rule, past whole lines above this one, then wherever the cursor sits
        # in the (possibly wrapped) line it is actually on.
        index, column = self._locate()
        offset = terminal.visible(self._prefixes(prompt)[index]) + column
        caret_row = 1 + sum(counts[:index]) + offset // width
        total = 1 + sum(counts) + 1 + (len(rows) - 2 - len(counts))
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

    def _submit(self, prompt: str) -> str | None:
        """Leave the finished line in the scrollback and hand it back.

        The rules and the footer are chrome for the moment you are typing; kept
        in the transcript they would say things about a line that has already
        been answered.
        """
        # A line ending in a backslash is a line still being written -- the
        # continuation every shell uses, and the one way to open a line that
        # needs no terminal support at all.
        if self.buffer.endswith("\\"):
            self.buffer = self.buffer[:-1] + "\n"
            self.cursor = len(self.buffer)
            return None

        line = self.buffer
        prefixes = self._prefixes(prompt)
        self.buffer, self.cursor, self.draft = "", 0, ""
        self._erase()
        for prefix, text in zip(prefixes, line.split("\n")):
            self.stdout.write(f"{prefix}{text}\n")
        self.stdout.flush()
        if line.strip():
            # The chip is what was on screen, so the chip is what history
            # replays; expanding it here would make Up paste the whole file
            # back into a block that folded it for a reason.
            self.history.append(line)
        self.recall = len(self.history)
        return self._unfold(line)


#: A colour per mode, because the footer is read at a glance rather than read.
#: Orange is what the session already uses for the agent acting on its own, and
#: blue for the user's side of the exchange -- so auto wears one and plan the
#: other. Ask stays dim: it is the mode nothing has been changed away from.
COLOURS = {
    "ask": DIM,
    "auto": terminal.ORANGE,
    "plan": terminal.ACCENT,
}


def footer(mode: str, cycles: bool = True, stream=None) -> str:
    """The line under the prompt: what mode this is, and how to change it."""
    colour = COLOURS.get(mode, DIM)
    text = f"{mode} mode on"
    if cycles:
        text += " (shift+tab to cycle)"
    return f"{terminal.paint('▸▸', colour, stream)} {terminal.paint(text, colour, stream)}"


#: How wide the context meter is drawn. Twelve cells is a percentage point
#: about every eight, which is as much as a bar read at a glance is worth.
METER = 12

#: The share of the window above which the meter changes colour. The same
#: number `context.COMPACT_AT` compacts at, so the row goes warm exactly when
#: the next turn is the one that will summarise -- and not a moment before,
#: which would be a warning about nothing.
WARM_AT = 0.75


def tokens(count: int) -> str:
    """``840`` or ``12.4k``, whichever is shorter to read.

    Thousands rather than exact counts past a thousand: nobody acts on the last
    three digits of a context window, and a field that changes width every turn
    makes the row jitter under the cursor.
    """
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k".replace(".0k", "k")
    return f"{count / 1_000_000:.1f}M".replace(".0M", "M")


def status(
    name: str = "",
    *,
    branch: str = "",
    ratio: float = 0.0,
    model: str = "",
    spent: int = 0,
    note: str = "",
    width: int = 80,
    stream=None,
) -> str:
    """The row above the mode line: where you are, and what the turn has cost.

    Every field is passed in rather than looked up. The row is rebuilt on each
    redraw -- which is each keystroke -- so anything it did for itself, a
    subprocess or a walk of the history, it would do a hundred times a line.

    Fields are dropped from the right as the terminal narrows, in the order
    below, because an eighty-column window cannot hold all of them and a row
    that wrapped would break the caret arithmetic in :meth:`Editor._wrapped`.
    What survives to the last is the directory and the branch: they say which
    checkout is about to be edited, which is the one thing here that can make a
    keystroke a mistake.
    """
    if not name:
        return ""
    # Ordered by what is worth reading first, which is also the reverse of the
    # order they are dropped in: where you are, then how much room is left,
    # then what is answering, then what it has spent.
    left: list[tuple[str, str]] = [(name, terminal.BOLD)]
    if branch:
        left.append((f"({branch})", terminal.ACCENT))
    if ratio or spent:
        meter = terminal.bar(ratio, METER, stream)
        colour = terminal.ORANGE if ratio >= WARM_AT else DIM
        left.append((f"ctx {meter} {round(ratio * 100)}% used", colour))
    if model:
        left.append((f"[{model}]", DIM))
    if spent:
        left.append((f"{tokens(spent)} tokens", DIM))

    #: How many of those survive to the last: the name, the branch, and the
    #: meter. The first two say which checkout is about to be edited and the
    #: third says whether the next turn has room to do it -- the three things
    #: here that change what a keystroke means.
    KEPT = 3

    # Measured plain and painted afterwards: `terminal.fit` hands coloured text
    # back whole rather than cutting an escape in half, so a row assembled in
    # colour could not be trimmed at all.
    #
    # Dropped from the end, one at a time, because the alternative -- fitting
    # the finished row -- cuts a field mid-word and leaves "ctx ███░░ 4" on
    # screen, which reads as a number rather than as something missing.
    def plain(parts: list[tuple[str, str]]) -> str:
        return "  ".join(text for text, _ in parts)

    reserved = terminal.visible(note) + 1 if note else 0
    while len(plain(left)) + reserved > width - 1:
        if len(left) > KEPT:
            left.pop()
        elif note:
            # The note goes once the spend and the model have: it is news about
            # a release, and a terminal this narrow should spend its columns on
            # the session in front of it rather than on the next one.
            note, reserved = "", 0
        elif len(left) > 1:
            left.pop()
        else:
            # Nothing left to drop. The name alone is wider than the terminal,
            # so it is cut -- the only case in this row where a field is.
            left = [(terminal.fit(plain(left), width - 1), left[0][1])]
            break

    row = "  ".join(terminal.paint(text, colour, stream) for text, colour in left)
    if not note:
        return f" {row}"
    pad = max(1, width - 1 - len(plain(left)) - terminal.visible(note))
    return f" {row}{' ' * pad}{terminal.paint(note, DIM, stream)}"


def footer_rows(mode: str, cycles: bool = True, stream=None, **fields) -> str:
    """The whole block under the input: the status row, then the mode row.

    One string with a newline in it rather than a list, so that the callable
    :class:`Editor` already takes needs no new shape -- and so that a caller
    with nothing to say about the workspace can keep returning one row.
    """
    rows = [row] if (row := status(stream=stream, **fields)).strip() else []
    rows.append(footer(mode, cycles, stream))
    return "\n".join(rows)
