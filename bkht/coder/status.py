"""The line that shows the turn is still alive.

A local 14b model can think for half a minute before it emits anything. With
no feedback that is indistinguishable from a hang, and the honest reaction is
to press Ctrl-C -- so the spinner is not decoration, it is what stops people
killing turns that were working.

Everything is injected (writer, clock, the enabled flag) so the frames can be
asserted exactly in tests, with no terminal and no sleeping.
"""

from __future__ import annotations

import contextlib
import threading
import time

from .terminal import (
    CLEAR_LINE,
    CURSOR_DOWN,
    CURSOR_UP,
    DIM,
    ERASE_BELOW,
    HIDE_CURSOR,
    RESET,
    SHOW_CURSOR,
    width,
)

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
INTERVAL = 0.1


class Status:
    """A single line, rewritten in place, while a turn runs."""

    def __init__(self, writer=None, clock=time.monotonic, enabled: bool = True,
                 interval: float = INTERVAL, colour: bool = True,
                 cancellable: bool = False, block=None) -> None:
        self.writer = writer
        self.clock = clock
        self.enabled = enabled
        self.interval = interval
        self.colour = colour
        #: Whether Esc will stop this turn -- said on the line, because a key
        #: nobody is told about is a key nobody presses.
        self.cancellable = cancellable
        #: Rows pinned under the spinner for as long as it draws: the prompt
        #: block, so the session keeps its shape while a turn runs instead of
        #: emptying out to one line and filling back in afterwards.
        #:
        #: A callable, and asked on every frame, because what it says -- the
        #: mode, the window, the token count -- is what the turn is changing.
        #: Rows come painted and pre-fitted; this only stacks them.
        self.block = block

        self.label = "working"
        self.tokens = 0
        self._started: float | None = None
        self._step = 0
        self._drawn = 0  # how many rows the last paint left on screen
        self._pinned: list[str] = []  # the block rows as they were last painted
        self._column = 0  # columns of half-written prose on the line above
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Re-entrant because pause() may nest: an approval prompt inside a
        # tool-call line inside a turn.
        self._lock = threading.RLock()

    # --- lifecycle ----------------------------------------------------------

    def start(self, label: str = "working") -> None:
        if not self.enabled or self._thread is not None:
            return
        self.label = label
        self.tokens = 0
        self._started = self.clock()
        self._step = 0
        self._stop.clear()
        self._write(HIDE_CURSOR)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Erase the line and join the thread. Safe to call twice."""
        thread, self._thread = self._thread, None
        if thread is None and self._started is None:
            return
        self._stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        with self._lock:
            self._erase()
            self._write(SHOW_CURSOR)
        self._started = None

    def __enter__(self) -> "Status":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- updates ------------------------------------------------------------

    def inline(self, column: int) -> None:
        """Say how much unfinished prose sits on the line the cursor is on.

        Zero means the last thing written ended in a newline, so this owns the
        line it is standing on. Anything else means a sentence is half-written
        there: the block moves down a row to leave it alone, and the cursor is
        put back at ``column`` afterwards so the next token carries on where the
        last one stopped.

        Before this the answer simply took the block off the screen -- prose
        arrives a fragment at a time and almost never ends on a newline, so
        "while the model is answering" was most of a turn.
        """
        with self._lock:
            if column != self._column and self._drawn:
                self._erase()
            self._column = max(0, column)

    def note(self, label: str) -> None:
        """Change what the line says without restarting the clock."""
        self.label = label

    def add_tokens(self, count: int = 1) -> None:
        self.tokens += count

    @contextlib.contextmanager
    def pause(self):
        """Clear the line for the duration, then put it back.

        Every other writer goes through this. Holding the lock across the yield
        is the point: the drawing thread cannot repaint into the middle of
        whatever the caller is printing.
        """
        with self._lock:
            self._erase()
            try:
                yield
            finally:
                if self._thread is not None:
                    self._draw()

    # --- rendering ----------------------------------------------------------

    def elapsed(self) -> float:
        return 0.0 if self._started is None else max(0.0, self.clock() - self._started)

    def frame(self) -> str:
        """The line as it would be drawn now, without terminal control codes."""
        spinner = FRAMES[self._step % len(FRAMES)]
        parts = [f"  {spinner} {self.label}…"]
        seconds = int(self.elapsed())
        if seconds:
            parts.append(f"{seconds}s")
        if self.tokens:
            parts.append(f"{self.tokens:,} tok")
        if self.cancellable:
            parts.append("esc to stop")
        line = " · ".join(parts)
        limit = max(1, width() - 1)
        return line if len(line) <= limit else line[: limit - 1] + "…"

    def _spin(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._draw()
                self._step += 1
            self._stop.wait(self.interval)

    def rows(self) -> list[str]:
        """Every row this owns: the spinner, and whatever is pinned below it.

        Below rather than above, so the turn's output scrolls up out of a block
        that stays where it is -- and so the last thing on screen is the prompt,
        which is where the eye goes back to when the turn ends.
        """
        text = self.frame()
        spinner = f"{DIM}{text}{RESET}" if self.colour else text
        return [spinner, *(self.block() if self.block is not None else [])]

    def _draw(self) -> None:
        spinner, *block = self.rows()

        # A sentence half-written on this line and nothing pinned to keep on
        # screen: the spinner has nothing to add that the arriving prose does
        # not already say, and a row of it under every fragment is noise. This
        # is what a one-shot run and a session with no block still do.
        if self._column and not block:
            return

        # Ten times a second, and the block below is the same ten times out of
        # ten -- the spinner is the only row with a new frame in it. Repainting
        # all six was six rows of flicker for one row of news, so the usual
        # frame reaches for the spinner's row alone and comes back.
        #
        # Where it reaches from is the whole of the arithmetic: the cursor rests
        # on the half-written prose line when there is one, and on the bottom
        # row of the block when there is not. Walking up from the wrong one of
        # those painted the spinner over the answer.
        if self._drawn == len(block) + 1 and block == self._pinned:
            if self._column:
                self._write(f"{CURSOR_DOWN}{CLEAR_LINE}{spinner}{CURSOR_UP}{self._back()}")
            else:
                self._write(
                    f"{self._up(len(block))}{CLEAR_LINE}{spinner}{self._down(len(block))}\r"
                )
            return

        # Back to the top of what is already there. A one-line spinner erased
        # itself -- every paint began with a line clear on the row it stood on
        # -- and a block cannot: the cursor rests at the bottom of it, so
        # painting from there stranded the rows above and drew a second block
        # under them. It happened whenever the block's own text changed, which
        # is every time the token count ticks.
        self._erase()

        rows = [spinner, *block]
        # A newline first when a sentence is half-written above: it is the row
        # this would otherwise paint over. Everything after moves relative to
        # where the cursor already is, which is what makes it survive the scroll
        # that drawing at the bottom of the screen causes.
        lead = "\n" if self._column else ""
        self._write(lead + "\n".join(f"{CLEAR_LINE}{row}" for row in rows))
        self._drawn = len(rows)
        self._pinned = block
        if self._column:
            self._write(self._up(len(rows)) + self._back())

    def _back(self) -> str:
        """Back to where the prose stopped, so the next token carries on there.

        Modulo the terminal, because a sentence longer than the screen is
        already several rows: the caller counts the columns it has written and
        cannot know where they wrapped. Asked for column 148 of a 145-column
        terminal, the cursor stops at the right-hand edge, and every token after
        that arrives with a run of spaces in front of it.
        """
        column = self._column % max(1, width())
        return f"\r\033[{column}C" if column else "\r"

    @staticmethod
    def _up(rows: int) -> str:
        return "" if rows <= 0 else (CURSOR_UP if rows == 1 else f"\033[{rows}A")

    @staticmethod
    def _down(rows: int) -> str:
        return "" if rows <= 0 else (CURSOR_DOWN if rows == 1 else f"\033[{rows}B")

    def _erase(self) -> None:
        """Take back every row the last paint left, ending where it started.

        Walked up one row at a time rather than jumped: the cursor has to end
        on the first of them, because that is the line the next writer -- prose,
        a tool line, an approval -- carries on from.
        """
        if not self._drawn:
            return
        if self._column:
            # The cursor is on the half-written line above the block. Step down
            # onto the block, take all of it, and come back to the prose.
            self._write(f"{CURSOR_DOWN}{ERASE_BELOW}{CURSOR_UP}{self._back()}")
        else:
            self._write(self._up(self._drawn - 1) + ERASE_BELOW)
        self._drawn = 0
        self._pinned = []

    def _write(self, text: str) -> None:
        if not self.enabled or self.writer is None:
            return
        try:
            self.writer.write(text)
            self.writer.flush()
        except (OSError, ValueError):
            # The terminal went away mid-turn; losing the spinner is not worth
            # losing the turn over.
            self.enabled = False


def for_stream(stream, enabled: bool) -> Status:
    """A status line bound to ``stream``, or an inert one when not a terminal."""
    return Status(writer=stream, enabled=enabled)
