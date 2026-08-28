"""Esc, while a turn is running, means stop.

Ctrl-C already abandons a turn, and it will keep working. It is the wrong key
to have to reach for: a local 14b model can spend minutes on a turn that went
wrong in its first ten seconds, and the reflex that key trains -- lean on it,
lean harder when nothing happens -- is the reflex that kills sessions people
meant to keep.

So the terminal is read while the turn runs, by a thread of its own, and a bare
Esc raises ``KeyboardInterrupt`` in the thread doing the work. That lands in the
handler Ctrl-C has always landed in, which is the point: one way to abandon a
turn, two keys that reach it.

Two things this cannot do. Esc arrives as the first byte of every arrow key, so
a bare Esc is only known to be bare once nothing follows it -- there is a short
wait built in, and it is why an arrow key pressed mid-turn is swallowed rather
than acted on. And the interrupt is raised, not delivered: the work thread
takes it at its next bytecode boundary, so a turn blocked in a socket read
stops when the next token arrives rather than instantly.
"""

from __future__ import annotations

import contextlib
import threading
import _thread

ESC = "\x1b"

#: How long to wait after an Esc for the rest of an escape sequence. Long
#: enough that an arrow key's own bytes have arrived on any terminal worth
#: supporting, short enough that a deliberate Esc still feels like a keypress.
SEQUENCE_WAIT = 0.05

#: How long the reader blocks before looking at its own flags again. It bounds
#: how long `pause` waits to take the terminal back, so it is small.
POLL = 0.05


def available(stdin=None, stdout=None) -> bool:
    """True when the terminal can be read a keystroke at a time."""
    from . import lineedit

    return lineedit.available(stdin, stdout)


class Watch:
    """Reads the terminal while a turn runs, so Esc can stop it.

    ``interrupt`` is injected so the tests can assert the cancellation without
    a signal reaching the process running them.
    """

    def __init__(self, stdin=None, interrupt=None, enabled: bool = True) -> None:
        import sys

        self.stdin = sys.stdin if stdin is None else stdin
        self.interrupt = interrupt or _thread.interrupt_main
        self.enabled = enabled
        self.cancelled = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Held for as long as the terminal belongs to somebody: the reader
        # while it selects, `pause` for the whole of an approval prompt.
        self._lock = threading.RLock()

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        try:
            import termios
            import tty

            self._fd = self.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            # No terminal to read, so nothing to watch. The turn still runs;
            # it just cannot be stopped with Esc.
            self.enabled = False
            return
        self.cancelled = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        try:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        except Exception:
            pass

    @contextlib.contextmanager
    def watching(self):
        """Watch for the length of one turn."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    @contextlib.contextmanager
    def pause(self):
        """Hand the terminal back for an approval prompt.

        Taken as a lock rather than a flag, so the reader cannot be halfway
        into a read when the prompt starts: the keypress that answers a
        question about a diff must not be eaten by the thread watching for Esc.
        """
        if self._thread is None:
            yield
            return
        with self._lock:
            yield

    # --- reading ------------------------------------------------------------

    def _read(self) -> None:
        import select

        while not self._stop.is_set():
            with self._lock:
                if not self._pending(select, POLL):
                    continue
                try:
                    key = self.stdin.read(1)
                except Exception:
                    return
                if key != ESC:
                    continue
                # An arrow key is an Esc with a tail. Wait for one; if none
                # comes, the Esc was meant on its own.
                if self._pending(select, SEQUENCE_WAIT):
                    self._swallow(select)
                    continue
            self.cancelled = True
            self.interrupt()
            return

    def _pending(self, select, timeout: float) -> bool:
        try:
            ready, _, _ = select.select([self._fd], [], [], timeout)
        except Exception:
            return False
        return bool(ready)

    def _swallow(self, select) -> None:
        """Read the rest of an escape sequence, so its tail is not typed."""
        while self._pending(select, 0.0):
            try:
                self.stdin.read(1)
            except Exception:
                return
