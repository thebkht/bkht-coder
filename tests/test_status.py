"""The live status line: frames, pausing, and the inert non-terminal path."""

from __future__ import annotations

import io

from bkht.coder.status import FRAMES, Status
from bkht.coder.terminal import CLEAR_LINE


class Clock:
    """A clock that only moves when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def build(enabled: bool = True) -> tuple[Status, io.StringIO, Clock]:
    writer, clock = io.StringIO(), Clock()
    return Status(writer=writer, clock=clock, enabled=enabled, colour=False), writer, clock


def test_the_frame_carries_elapsed_time_and_tokens():
    status, _, clock = build()
    status.start("thinking")
    clock.now = 12.4
    status.add_tokens(1847)

    frame = status.frame()
    assert "thinking…" in frame
    assert "12s" in frame
    assert "1,847 tok" in frame
    status.stop()


def test_the_frame_omits_counters_that_are_still_zero():
    # A turn that has just begun should read "working…", not "working… 0s · 0 tok".
    # The spinner glyph itself rotates on a thread, so it is not asserted on.
    status, _, _ = build()
    status.start()
    glyph, _, rest = status.frame().strip().partition(" ")
    assert glyph in FRAMES
    assert rest == "working…"
    status.stop()


def test_a_disabled_status_writes_nothing_at_all():
    status, writer, _ = build(enabled=False)
    status.start()
    status.add_tokens(10)
    with status.pause():
        pass
    status.stop()
    assert writer.getvalue() == ""


def test_pause_clears_the_line_before_yielding():
    status, writer, _ = build()
    status.start()
    status._draw()
    writer.seek(0)
    writer.truncate()

    with status.pause():
        assert writer.getvalue() == CLEAR_LINE, "the line must be gone before the caller prints"
    status.stop()


def test_stop_erases_the_line_and_is_safe_twice():
    status, writer, _ = build()
    status.start()
    status._draw()
    status.stop()
    status.stop()
    assert writer.getvalue().endswith(CLEAR_LINE + "\033[?25h")


def test_the_line_is_truncated_to_the_terminal_width(monkeypatch):
    monkeypatch.setattr("bkht.coder.status.width", lambda: 20)
    status, _, _ = build()
    status.start("a very long label indeed that will not fit")
    assert len(status.frame()) <= 19
    status.stop()


def test_a_dead_writer_disables_the_status_rather_than_raising():
    writer = io.StringIO()
    status = Status(writer=writer, clock=Clock(), colour=False)
    status.start()
    writer.close()
    status._draw()  # must not raise
    assert not status.enabled


def test_the_line_offers_esc_when_the_turn_can_be_cancelled():
    line = Status(writer=None, enabled=False, cancellable=True)
    assert "esc to stop" in line.frame()


def test_the_line_stays_quiet_about_esc_when_it_would_not_work():
    line = Status(writer=None, enabled=False)
    assert "esc" not in line.frame()
