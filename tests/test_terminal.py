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
