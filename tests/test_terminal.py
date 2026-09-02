"""Terminal capability detection and single-key reads."""

from __future__ import annotations

import io

from bkht.coder import terminal


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def test_interactive_requires_both_ends():
    assert terminal.interactive(FakeTTY(), FakeTTY())
    assert not terminal.interactive(FakeTTY(), io.StringIO())
    assert not terminal.interactive(io.StringIO(), FakeTTY())


def test_interactive_survives_a_closed_stream():
    closed = io.StringIO()
    closed.close()
    assert not terminal.interactive(closed, FakeTTY())


def test_paint_colours_only_a_terminal():
    assert terminal.paint("hi", terminal.DIM, FakeTTY()) == f"{terminal.DIM}hi{terminal.RESET}"
    assert terminal.paint("hi", terminal.DIM, io.StringIO()) == "hi"


def test_read_key_falls_back_to_a_line_when_not_a_terminal():
    # A pipe cannot be put into cbreak, so the whole line is read and its first
    # character answers the prompt -- `y<Enter>` still works under `echo y |`.
    assert terminal.read_key(io.StringIO("y\n")) == "y"
    assert terminal.read_key(io.StringIO("always\n")) == "a"


def test_read_key_returns_empty_on_eof():
    assert terminal.read_key(io.StringIO("")) == ""


def test_width_has_a_default():
    assert terminal.width() > 0


# --- the accent palette -----------------------------------------------------


def test_a_256_colour_terminal_is_recognised_from_term():
    assert terminal.supports_256({"TERM": "xterm-256color"})


def test_truecolor_implies_256_even_without_it_in_term():
    assert terminal.supports_256({"TERM": "xterm", "COLORTERM": "truecolor"})


def test_a_plain_terminal_does_not_claim_256_colours():
    assert not terminal.supports_256({"TERM": "xterm"})
    assert not terminal.supports_256({})


def test_the_accents_are_sequences_paint_can_strip_again():
    # visible() has to measure a coloured accent as nothing, or every column
    # drawn beside one lands short by the width of the escape.
    for accent in (terminal.ORANGE, terminal.ACCENT):
        assert terminal.visible(f"{accent}word{terminal.RESET}") == 4


def test_a_bar_fills_the_share_it_is_given():
    assert terminal.bar(0.5, 10) == "█████░░░░░"
    assert terminal.bar(0.0, 4) == "░░░░"
    assert terminal.bar(1.0, 4) == "████"


def test_a_bar_that_has_started_never_reads_as_empty():
    # 1% of a twelve-cell bar rounds to nothing, and a meter that shows nothing
    # while the window fills is a meter nobody would trust the rest of the time.
    assert terminal.bar(0.01, 12).startswith("█")


def test_a_bar_past_full_stops_at_full():
    # The window can be over budget between the turn that filled it and the
    # compaction that follows; the row must not grow a column when it is.
    assert terminal.bar(1.4, 6) == "██████"


def test_a_bar_falls_back_to_ascii_where_blocks_cannot_be_written():
    # A Windows console on cp1252 raises on the block glyphs rather than
    # drawing them badly, so the meter has to be askable in Latin-1 too.
    class Console:
        encoding = "cp1252"

        def isatty(self):
            return True

    assert terminal.bar(0.5, 4, stream=Console()) == "##--"


def test_encodable_answers_for_the_stream_it_is_given():
    class Utf8:
        encoding = "utf-8"

    class Legacy:
        encoding = "cp1252"

    class Silent:
        encoding = None

    assert terminal.encodable("\u2588", Utf8())
    assert not terminal.encodable("\u2588", Legacy())
    assert not terminal.encodable("\u2588", Silent())
