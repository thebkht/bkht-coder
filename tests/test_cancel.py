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


def test_ordinary_typing_is_not_a_cancellation(terminal):
    made, fired = watch(terminal)
    terminal.press("hello there")
    assert not fired.wait(timeout=0.3)
    assert made.cancelled is False
    settle(made)


def test_ordinary_typing_is_kept_for_the_next_prompt(terminal):
    """It used to be read and dropped.

    This thread owns the terminal for the length of a turn, so a key pressed
    during one arrives here and the loop's own read is not running to receive
    it. A user who typed their next question while waiting watched it vanish.
    """
    made, fired = watch(terminal)
    terminal.press("what about the tests")
    time.sleep(0.3)
    settle(made)
    assert made.typed() == "what about the tests"


def test_taking_the_type_ahead_empties_it(terminal):
    """A draft handed to one prompt must not turn up in the next one too."""
    made, _ = watch(terminal)
    terminal.press("once")
    time.sleep(0.3)
    settle(made)
    assert made.typed() == "once"
    assert made.typed() == ""


def test_enter_is_not_carried_over(terminal):
    """It was meant to send a message nothing was reading.

    Carried over, it would send whatever had been typed the instant the turn
    ended -- without the user having seen a word of it.
    """
    made, _ = watch(terminal)
    terminal.press("send this\r\n")
    time.sleep(0.3)
    settle(made)
    assert made.typed() == "send this"


def test_an_arrow_key_leaves_nothing_behind(terminal):
    """Its tail is a terminal instruction, not something anybody typed."""
    made, _ = watch(terminal)
    terminal.press("\x1b[A")
    time.sleep(0.3)
    settle(made)
    assert made.typed() == ""


def test_the_type_ahead_is_bounded(terminal):
    """This reads whatever the terminal hands over, redirected files included."""
    made, _ = watch(terminal)
    terminal.press("x" * (cancel.TYPEAHEAD_LIMIT + 500))
    time.sleep(0.5)
    settle(made)
    assert len(made.typed()) == cancel.TYPEAHEAD_LIMIT


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


# --- interruptible --------------------------------------------------------


def test_interruptible_yields_what_the_source_yields():
    assert list(cancel.interruptible(iter([1, 2, 3]))) == [1, 2, 3]


def test_an_error_in_the_source_is_raised_on_the_asking_thread():
    def angry():
        yield 1
        raise ValueError("no")

    got = []
    with pytest.raises(ValueError):
        for item in cancel.interruptible(angry()):
            got.append(item)
    assert got == [1]


def test_a_blocked_source_does_not_block_the_caller():
    # The bug this exists for: a turn waiting on a socket for a 14b's first
    # token ran no bytecode, so the Esc that had already been pressed was not
    # delivered until the model spoke. The caller has to keep reaching a
    # bytecode boundary while the read blocks.
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(5)
        yield "late"

    stream = cancel.interruptible(slow(), poll=0.01)
    hops = []

    def count():
        # Stands in for the interpreter noticing an interrupt: it can only run
        # between the caller's own bytecodes.
        started.wait(2)
        for _ in range(3):
            hops.append(time.monotonic())
            time.sleep(0.02)
        release.set()

    watcher = threading.Thread(target=count)
    watcher.start()
    assert list(stream) == ["late"]
    watcher.join(2)
    assert len(hops) == 3


def test_abandoning_the_stream_stops_the_worker_reading():
    # A cancelled turn leaves the generator unfinished. The worker must not go
    # on draining a reply nobody is listening to.
    read = []

    def endless():
        # Paced, so the worker is still reading when the caller lets go --
        # a source that ran to the end first would prove nothing.
        for number in range(1000):
            read.append(number)
            yield number
            time.sleep(0.01)

    stream = cancel.interruptible(endless(), poll=0.01)
    assert next(stream) == 0
    stream.close()
    time.sleep(0.1)
    settled = len(read)
    time.sleep(0.1)
    assert len(read) == settled, "the worker went on reading after being let go"
    assert settled < 1000
