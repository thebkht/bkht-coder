"""The live status line: frames, pausing, and the inert non-terminal path."""

from __future__ import annotations

import io

from bkht.coder.status import FRAMES, Status
from bkht.coder.terminal import CURSOR_DOWN, CURSOR_UP, ERASE_BELOW


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
        # Erased to the bottom of the screen rather than line by line: the
        # spinner may be pinning a block of rows under itself, and one code
        # takes back all of them however they wrapped.
        assert writer.getvalue() == ERASE_BELOW, "the line must be gone before the caller prints"
    status.stop()


def test_stop_erases_the_line_and_is_safe_twice():
    status, writer, _ = build()
    status.start()
    status._draw()
    status.stop()
    status.stop()
    assert writer.getvalue().endswith(ERASE_BELOW + "\033[?25h")


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


# --- the pinned block -----------------------------------------------------


def test_the_spinner_pins_the_rows_it_is_given_under_itself():
    line = Status(writer=io.StringIO(), clock=iter([0, 0, 0]).__next__, colour=False)
    line.block = lambda: ["one", "two"]
    rows = line.rows()
    # Under, not over: the turn's output scrolls up out of a block that stays,
    # and the prompt is the last thing on screen when the turn ends.
    assert rows[0].endswith("working…")
    assert rows[1:] == ["one", "two"]


def test_a_pinned_block_is_taken_back_whole():
    writer = io.StringIO()
    line = Status(writer=writer, clock=iter([0, 0, 0, 0]).__next__, colour=False, interval=0)
    line.block = lambda: ["one", "two", "three"]
    line._draw()
    writer.truncate(0), writer.seek(0)
    line._erase()
    # Three rows below the spinner means the cursor walks up three to reach the
    # row the next writer carries on from.
    assert writer.getvalue() == "\033[3A" + ERASE_BELOW


def test_a_block_that_changes_is_redrawn_as_it_is():
    # The rows say what the turn is changing -- the token count, the meter --
    # so they are asked for again on every frame rather than kept.
    counter = iter(["first", "second"])
    line = Status(writer=io.StringIO(), clock=lambda: 0, colour=False)
    line.block = lambda: [next(counter)]
    assert line.rows()[1] == "first"
    assert line.rows()[1] == "second"


def test_no_block_is_the_line_on_its_own():
    line = Status(writer=io.StringIO(), clock=lambda: 0, colour=False)
    assert len(line.rows()) == 1


def test_the_block_stays_while_a_sentence_is_half_written():
    # The bug this exists for: prose arrives a fragment at a time and almost
    # never ends on a newline, so hiding the block whenever one was pending
    # meant hiding it for the whole of an answer.
    writer = io.StringIO()
    line = Status(writer=writer, clock=lambda: 0, colour=False)
    line.block = lambda: ["pinned"]
    line.inline(19)
    line._draw()
    painted = writer.getvalue()
    assert "pinned" in painted
    # Below the prose, not over it: a newline opens the row it draws into.
    assert painted.startswith("\n")
    # And the cursor is put back where the sentence stopped, so the next token
    # carries on from there rather than under the block.
    assert painted.endswith("\r\033[19C")


def test_a_bare_spinner_still_keeps_off_a_half_written_line():
    # With nothing pinned there is nothing worth a second row: the prose is
    # already saying the turn is alive.
    writer = io.StringIO()
    line = Status(writer=writer, clock=lambda: 0, colour=False)
    line.inline(12)
    line._draw()
    assert writer.getvalue() == ""


def test_the_block_is_taken_back_from_under_the_prose():
    writer = io.StringIO()
    line = Status(writer=writer, clock=lambda: 0, colour=False)
    line.block = lambda: ["pinned"]
    line.inline(19)
    line._draw()
    writer.truncate(0), writer.seek(0)
    line._erase()
    # Down onto the block, take it, and come back to where the prose stopped.
    assert writer.getvalue() == CURSOR_DOWN + ERASE_BELOW + CURSOR_UP + "\r\033[19C"


def test_the_quick_repaint_reaches_from_wherever_the_cursor_rests():
    # The cursor rests on the half-written prose line when there is one, and on
    # the bottom row of the block when there is not. Reaching up from the wrong
    # one of those painted the spinner over the answer.
    writer = io.StringIO()
    line = Status(writer=writer, clock=lambda: 0, colour=False)
    line.block = lambda: ["pinned"]
    line.inline(19)
    line._draw()          # the full paint
    writer.truncate(0), writer.seek(0)
    line._draw()          # the quick one
    painted = writer.getvalue()
    assert painted.startswith(CURSOR_DOWN)   # down onto the spinner, not up
    assert painted.endswith("\r\033[19C")    # and back to the prose
    assert "pinned" not in painted           # the block itself is left alone


def test_a_changed_block_replaces_the_one_on_screen():
    # A one-line spinner erased itself: every paint began with a line clear on
    # the row it stood on. A block cannot -- the cursor rests at the bottom of
    # it -- so painting from there stranded the rows above and drew a second
    # block under them. It fired whenever the block's text changed, which is
    # every time the token count ticks.
    writer = io.StringIO()
    rows = ["first"]
    line = Status(writer=writer, clock=lambda: 0, colour=False)
    line.block = lambda: list(rows)
    line._draw()
    rows[:] = ["second"]
    writer.truncate(0), writer.seek(0)
    line._draw()
    painted = writer.getvalue()
    assert painted.startswith(CURSOR_UP + ERASE_BELOW), "the old block must go first"
    assert "second" in painted
    assert line._drawn == 2
