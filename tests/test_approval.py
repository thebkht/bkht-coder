"""Single-key approval: what each key means, and what silence means."""

from __future__ import annotations

import io

from bkht.coder.approval import ask_tty

BIG = "\n".join(f"+line {i}" for i in range(200))


def keys(*presses):
    """A key source that replays ``presses``, then behaves like a closed tty."""
    queue = list(presses)
    return lambda: queue.pop(0) if queue else ""


def test_each_key_maps_to_its_answer():
    for key, expected in [("y", "y"), ("n", "n"), ("a", "a"), ("Y", "y"), ("A", "a")]:
        assert ask_tty("Allow?", "body", keys=keys(key), out=io.StringIO()) == expected


def test_anything_unrecognised_is_a_refusal():
    # Enter, an escape sequence, or a stream that closed under us. Silence must
    # not approve a write.
    for key in ["\n", "\x1b", "q", ""]:
        assert ask_tty("Allow?", "body", keys=keys(key), out=io.StringIO()) == "n"


def test_the_preview_is_bounded_until_d_expands_it():
    out = io.StringIO()
    assert ask_tty("Allow?", BIG, keys=keys("d", "y"), out=out) == "y"
    shown = out.getvalue()
    assert "more diff lines" in shown, "the first render must be the bounded one"
    assert "+line 199" in shown, "d must reveal the whole diff"


def test_d_without_a_body_is_simply_a_refusal():
    # Nothing to expand, so it must not spin waiting for another key.
    assert ask_tty("Allow?", "", keys=keys("d"), out=io.StringIO()) == "n"


def test_the_answer_is_echoed_into_the_scrollback():
    out = io.StringIO()
    ask_tty("Allow write_file?", "body", keys=keys("a"), out=out)
    assert "always" in out.getvalue()


def test_the_question_and_hint_are_shown():
    out = io.StringIO()
    ask_tty("Allow bash?", "$ rm -rf /", keys=keys("n"), out=out)
    shown = out.getvalue()
    assert "Allow bash?" in shown
    assert "$ rm -rf /" in shown
    assert "[y] yes" in shown
