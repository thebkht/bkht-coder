"""The logo, and the greeting that decides whether to draw it."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from bkht.coder import banner, cli, terminal


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal that can draw braille."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def bare(text: str) -> str:
    """The greeting with its colour taken back off, for comparing shapes."""
    return terminal.SGR.sub("", text)


def session(**kwargs):
    agent = SimpleNamespace(
        provider=SimpleNamespace(model="qwen2.5-coder:14b", num_ctx=8192),
        session=SimpleNamespace(prompt_tokens=1234),
    )
    permissions = SimpleNamespace(mode="ask")
    workspace = SimpleNamespace(root="/tmp/project")
    return agent, permissions, workspace


# --- the art ----------------------------------------------------------------


def test_the_logo_is_braille_and_nothing_else():
    # A stray space or block character would misalign every row beside it, and
    # only on the terminals that render the two at different widths.
    assert len(banner.LOGO) == 4
    for row in banner.LOGO:
        assert len(row) == banner.WIDTH
        assert all(0x2800 <= ord(dot) <= 0x28FF for dot in row)


def test_side_text_lands_on_the_rows_it_was_given():
    rows = banner.render([None, "name", None, "tagline"]).splitlines()
    assert rows[1].endswith("name")
    assert rows[3].endswith("tagline")
    assert rows[0] == banner.LOGO[0]


def test_side_text_past_the_last_row_carries_on_below_the_art():
    rows = banner.render([None, None, None, None, "after"]).splitlines()
    assert len(rows) == 5
    assert rows[4] == " " * (banner.WIDTH + banner.GUTTER) + "after"


def test_a_row_without_side_text_has_no_trailing_space():
    for row in banner.render(["only the first"]).splitlines():
        assert row == row.rstrip()


def test_a_stream_that_cannot_carry_braille_is_not_drawable():
    assert banner.drawable(SimpleNamespace(encoding="utf-8"))
    assert not banner.drawable(SimpleNamespace(encoding="ascii"))
    assert not banner.drawable(SimpleNamespace(encoding=None))


# --- the greeting -----------------------------------------------------------


def test_a_pipe_gets_the_two_lines_it_always_did():
    agent, permissions, workspace = session()
    greeting = cli.greeting(agent, permissions, workspace, io.StringIO())
    assert greeting == (
        "coder · qwen2.5-coder:14b · ask · 1,234/8,192 ctx · /tmp/project\n"
        "/help for commands, /exit to leave."
    )


def drawn(monkeypatch, width=100, **kwargs):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: width)
    agent, permissions, workspace = session()
    return bare(cli.greeting(agent, permissions, workspace, FakeTTY(), **kwargs))


def test_the_greeting_is_the_mark_with_the_facts_beside_it(monkeypatch):
    rows = drawn(monkeypatch).splitlines()
    assert rows[0].startswith(banner.LOGO[0])
    assert "bkht.coder" in rows[0]
    assert "qwen2.5-coder:14b" in rows[1]
    assert "/tmp/project" in rows[2]


def test_the_greeting_says_nothing_that_goes_stale_while_the_session_runs(monkeypatch):
    # The mode and the context count used to be a fourth row here. A greeting
    # is scrollback: it went on saying "ask" however many times Shift+Tab was
    # pressed, and "0 ctx" however many turns had been taken. Both live on the
    # row under the prompt now, which is redrawn on the keypress that changes
    # them.
    greeting = drawn(monkeypatch)
    assert "ask" not in greeting
    assert "ctx" not in greeting


def test_one_shape_at_every_width_worth_drawing_at(monkeypatch):
    # There used to be a second, wider layout, which was a second thing that
    # had to stay true of the first.
    for width in (banner.MIN_WIDTH, 90, 200):
        assert drawn(monkeypatch, width=width).splitlines()[0] == drawn(monkeypatch).splitlines()[0]


def test_what_loaded_is_listed_under_the_art(monkeypatch):
    loaded = cli.Loaded("", "skills: tdd\n1 skill(s) skipped: broken.md")
    assert loaded.lines() == ["skills: tdd", "1 skill(s) skipped: broken.md"]
    rows = drawn(monkeypatch, loaded=loaded).splitlines()
    assert rows[-1].strip() == "1 skill(s) skipped: broken.md"


def test_nothing_loaded_adds_no_rows(monkeypatch):
    assert len(drawn(monkeypatch, loaded=cli.Loaded("", "")).splitlines()) == len(banner.LOGO)


def test_only_the_name_is_coloured(monkeypatch):
    # A column of accents is a column that keeps asking to be read; a greeting
    # is meant to be read once.
    agent, permissions, workspace = session()
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: 100)
    rows = cli.greeting(agent, permissions, workspace, FakeTTY()).splitlines()
    assert terminal.ACCENT in rows[0]
    assert all(terminal.ACCENT not in row for row in rows[1:])


@pytest.mark.parametrize("width", [40, banner.MIN_WIDTH - 1])
def test_a_narrow_terminal_falls_back_rather_than_wrapping(monkeypatch, width):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: width)
    agent, permissions, workspace = session()

    greeting = bare(cli.greeting(agent, permissions, workspace, FakeTTY()))
    assert "⣿" not in greeting
    assert greeting.startswith("coder · ")


# --- the rule between exchanges ---------------------------------------------


def test_the_divider_spans_the_window(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: 72)
    gap, rule = bare(cli.divider()).splitlines()
    # The blank line is what makes the rule a boundary rather than the last
    # line of the answer above it.
    assert gap == ""
    assert rule == banner.RULE * 72


def test_a_pipe_gets_no_divider(monkeypatch):
    # A rule between exchanges is for a screen being scrolled back through; in
    # a redirected transcript it is a line of noise every turn.
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: False)
    assert cli.divider() == ""


def test_the_hint_is_drawn_below_the_line_being_typed(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    stream = FakeTTY()
    typed = []

    class Reader:
        def read(self, prompt):
            typed.append(prompt)
            return "hi"

    assert cli.read_line(Reader(), "> ", stream) == "hi"
    drawn = stream.getvalue()
    # Written first, then walked back over: readline edits the row above a
    # line that is already on screen.
    assert drawn.startswith("\n")
    assert cli.HINT in drawn
    assert drawn.index(cli.HINT) < drawn.index(terminal.CURSOR_UP)
    # read_line paints the prompt now: the editor path needs the plain text,
    # so the colour cannot be put on by the caller any more.
    assert typed == [terminal.paint("> ", terminal.ACCENT + terminal.BOLD, stream)]


def test_the_hint_is_cleared_once_the_line_is_submitted(monkeypatch):
    # It is for the moment you are typing; left behind it would sit stranded
    # above an answer it says nothing about.
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    stream = FakeTTY()

    class Reader:
        def read(self, prompt):
            return "hi"

    cli.read_line(Reader(), "> ", stream)
    assert stream.getvalue().endswith(terminal.CLEAR_LINE)


def test_the_hint_is_cleared_even_when_the_reader_raises(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    stream = FakeTTY()

    class Reader:
        def read(self, prompt):
            raise EOFError

    with pytest.raises(EOFError):
        cli.read_line(Reader(), "> ", stream)
    assert stream.getvalue().endswith(terminal.CLEAR_LINE)


def test_a_pipe_gets_the_prompt_and_no_cursor_games(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: False)
    stream = io.StringIO()

    class Reader:
        def read(self, prompt):
            return "hi"

    assert cli.read_line(Reader(), "> ", stream) == "hi"
    assert stream.getvalue() == ""


def test_an_available_release_is_the_last_line_of_the_greeting(monkeypatch):
    rows = drawn(monkeypatch, notice="v0.3.0 available · coder update").splitlines()
    assert rows[-1].endswith("v0.3.0 available · coder update")


def test_a_pipe_gets_the_release_line_as_plain_text():
    # Off a terminal the greeting is two lines and always has been; a notice
    # goes between them rather than turning it into a box.
    agent, permissions, workspace = session()
    greeting = cli.greeting(
        agent, permissions, workspace, io.StringIO(),
        notice="v0.3.0 available · coder update",
    )
    lines = greeting.splitlines()
    assert lines[1] == "v0.3.0 available · coder update"
    assert lines[-1] == "/help for commands, /exit to leave."


def test_no_release_leaves_the_greeting_exactly_as_it_was(monkeypatch):
    # The empty string has to add nothing at all -- not a blank row in the box,
    # and not a blank line in the piped form.
    assert drawn(monkeypatch, notice="") == drawn(monkeypatch)

    agent, permissions, workspace = session()
    plain = cli.greeting(agent, permissions, workspace, io.StringIO())
    assert cli.greeting(agent, permissions, workspace, io.StringIO(), notice="") == plain
