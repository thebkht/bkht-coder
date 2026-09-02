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

One thing this cannot do. Esc arrives as the first byte of every arrow key, so
a bare Esc is only known to be bare once nothing follows it -- there is a short
wait built in, and it is why an arrow key pressed mid-turn is swallowed rather
than acted on.

The interrupt is raised, not delivered: the main thread takes it at its next
bytecode boundary. Blocked in a socket read it is running no bytecode at all,
and that used to mean Esc did nothing for the whole of the wait before the
first token -- which on a local 14b is half a minute, and is exactly the part
of a turn worth stopping. :func:`interruptible` is why it now lands: the read
happens on a worker, and the main thread waits in short hops it can be
interrupted between.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import _thread
from collections.abc import Iterable, Iterator

ESC = "\x1b"

#: How long to wait after an Esc for the rest of an escape sequence. Long
#: enough that an arrow key's own bytes have arrived on any terminal worth
#: supporting, short enough that a deliberate Esc still feels like a keypress.
SEQUENCE_WAIT = 0.05

#: How long the reader blocks before looking at its own flags again. It bounds
#: how long `pause` waits to take the terminal back, so it is small.
POLL = 0.05

#: What :func:`interruptible` puts on the queue to say the source is finished.
_DONE = object()


def interruptible(source: Iterable, poll: float = POLL) -> Iterator:
    """Yield ``source`` here, while something else does the waiting.

    ``interrupt_main`` sets a flag the main thread reads between bytecodes. A
    thread blocked in a socket read is between nothing, so an Esc pressed
    while the model thinks was remembered and then delivered whenever the first
    token happened to arrive -- which is to say, not when it was pressed.

    So the source is drained on a worker and handed over a queue, and the wait
    here is a series of short gets rather than one long read. Each get is
    bytecode, so the interrupt lands within ``poll`` of the key.

    The queue is unbounded on purpose. A bounded one would park the worker in
    ``put`` when this generator is abandoned mid-turn -- which is the ordinary
    end of a cancelled turn -- and a parked worker never reaches the ``finally``
    that closes the response. What accumulates instead is one reply's worth of
    chunks, which the window already bounds.

    An exception raised inside ``source`` is re-raised here, on the thread that
    asked for it, so callers see the failures they have always seen.
    """
    chunks: queue.Queue = queue.Queue()
    failure: list[BaseException] = []
    abandoned = threading.Event()

    def drain() -> None:
        try:
            for item in source:
                chunks.put(item)
                if abandoned.is_set():
                    break
        except BaseException as exc:  # re-raised below, on the caller's thread
            failure.append(exc)
        finally:
            # Closing the source is what releases the response behind it; the
            # generator's own `finally` does the work.
            close = getattr(source, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()
            chunks.put(_DONE)

    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = chunks.get(timeout=poll)
            except queue.Empty:
                continue
            if item is _DONE:
                break
            yield item
    finally:
        # Reached by an Esc, a Ctrl-C, or a caller that stopped early. The
        # worker is told so it can stop at its next chunk rather than reading
        # a reply nobody is listening to.
        abandoned.set()

    if failure:
        raise failure[0]


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
