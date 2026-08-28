"""Esc, while a turn runs, stops it.

The reader is a thread over a real file descriptor, so these drive it through
a pipe: write bytes into one end and assert what the watch made of them. The
interrupt is injected, so a cancellation here does not reach the process
running the tests.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from bkht.coder import cancel


class Terminal:
    """One end of a pipe, shaped like the bit of stdin the watch uses."""

    def __init__(self) -> None:
        self.read_fd, self.write_fd = os.pipe()
        self.stream = os.fdopen(self.read_fd, "rb", buffering=0)

    def fileno(self) -> int:
        return self.read_fd

    def read(self, count: int) -> str:
        return self.stream.read(count).decode("utf-8", "replace")

    def press(self, keys: str) -> None:
        os.write(self.write_fd, keys.encode())

    def close(self) -> None:
        for fd in (self.write_fd,):
            try:
                os.close(fd)
            except OSError:
                pass
        self.stream.close()


@pytest.fixture
def terminal():
    made = Terminal()
    yield made
    made.close()


def watch(terminal, **kwargs):
    """A watch over the pipe, with the terminal setup it cannot do here.

    `start` puts a real tty into cbreak, which a pipe is not; the reader
    thread is started directly so the rest of the class is under test.
    """
    fired = threading.Event()
    made = cancel.Watch(stdin=terminal, interrupt=fired.set, **kwargs)
    made._fd = terminal.fileno()
    made._thread = threading.Thread(target=made._read, daemon=True)
    made._thread.start()
    return made, fired


def settle(made) -> None:
    made._stop.set()
    if made._thread is not None:
        made._thread.join(timeout=1.0)


def test_a_bare_esc_cancels_the_turn(terminal):
    made, fired = watch(terminal)
    terminal.press("\x1b")
    assert fired.wait(timeout=2.0)
    assert made.cancelled is True
    settle(made)


def test_an_arrow_key_is_not_a_cancellation(terminal):
    # Esc is the first byte of every arrow key, so a watch that cancelled on
    # the byte alone would make the arrows unusable mid-turn.
    made, fired = watch(terminal)
    terminal.press("\x1b[A")
    assert not fired.wait(timeout=0.5)
    assert made.cancelled is False
    settle(made)


def test_ordinary_typing_is_ignored(terminal):
    made, fired = watch(terminal)
    terminal.press("hello there")
    assert not fired.wait(timeout=0.3)
    assert made.cancelled is False
    settle(made)


def test_an_esc_after_an_arrow_key_still_cancels(terminal):
    # The arrow's tail must be swallowed whole; a leftover byte would be read
    # as the next keypress and the Esc after it would go unnoticed.
    made, fired = watch(terminal)
    terminal.press("\x1b[B")
    time.sleep(0.2)
    terminal.press("\x1b")
    assert fired.wait(timeout=2.0)
    settle(made)


def test_pause_keeps_the_watch_off_the_terminal(terminal):
    # The keypress that answers an approval prompt has to reach the prompt,
    # not the thread waiting for Esc.
    made, fired = watch(terminal)
    with made.pause():
        terminal.press("y")
        time.sleep(0.2)
        assert terminal.read(1) == "y"
    settle(made)


def test_a_disabled_watch_starts_nothing():
    made = cancel.Watch(enabled=False)
    made.start()
    assert made._thread is None
    made.stop()  # must not raise


def test_pause_on_an_unstarted_watch_is_a_no_op():
    made = cancel.Watch(enabled=False)
    with made.pause():
        pass
