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
        self._suspended = False
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

    def suspend(self, suspended: bool = True) -> None:
        """Stop drawing while prose occupies the current line.

        The spinner repaints with a carriage return and a line clear, so
        sharing a line with half-written prose does not interleave -- it erases
        it. Silence is the only thing this line is allowed to fill.
        """
        with self._lock:
            if suspended and self._drawn:
                self._erase()
            self._suspended = suspended

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
        if self._suspended:
            return
        spinner, *block = self.rows()

        # Ten times a second, and the block below is the same ten times out of
        # ten -- the spinner is the only row with a new frame in it. Repainting
        # all six was six rows of flicker for one row of news, so the usual
        # frame reaches up for the spinner's row alone and steps back down.
        if self._drawn == len(block) + 1 and block == self._pinned:
            down = f"\033[{len(block)}B" if len(block) > 1 else CURSOR_DOWN * len(block)
            self._write(f"{CURSOR_UP * len(block)}{CLEAR_LINE}{spinner}{down}\r")
            return

        self._write("\n".join(f"{CLEAR_LINE}{row}" for row in [spinner, *block]))
        self._drawn = len(block) + 1
        self._pinned = block

    def _erase(self) -> None:
        """Take back every row the last paint left, ending where it started.

        Walked up one row at a time rather than jumped: the cursor has to end
        on the first of them, because that is the line the next writer -- prose,
        a tool line, an approval -- carries on from.
        """
        if not self._drawn:
            return
        self._write(CURSOR_UP * (self._drawn - 1) + ERASE_BELOW)
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
