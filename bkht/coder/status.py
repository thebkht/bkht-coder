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

from .terminal import CLEAR_LINE, DIM, HIDE_CURSOR, RESET, SHOW_CURSOR, width

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
INTERVAL = 0.1


class Status:
    """A single line, rewritten in place, while a turn runs."""

    def __init__(self, writer=None, clock=time.monotonic, enabled: bool = True,
                 interval: float = INTERVAL, colour: bool = True) -> None:
        self.writer = writer
        self.clock = clock
        self.enabled = enabled
        self.interval = interval
        self.colour = colour

        self.label = "working"
        self.tokens = 0
        self._started: float | None = None
        self._step = 0
        self._drawn = False
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
        line = " · ".join(parts)
        limit = max(1, width() - 1)
        return line if len(line) <= limit else line[: limit - 1] + "…"

    def _spin(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._draw()
                self._step += 1
            self._stop.wait(self.interval)

    def _draw(self) -> None:
        if self._suspended:
            return
        text = self.frame()
        self._write(f"{CLEAR_LINE}{DIM}{text}{RESET}" if self.colour else f"{CLEAR_LINE}{text}")
        self._drawn = True

    def _erase(self) -> None:
        if self._drawn:
            self._write(CLEAR_LINE)
            self._drawn = False

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
