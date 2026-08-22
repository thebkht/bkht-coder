"""The logo, and the greeting that decides whether to draw it."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from bkht.coder import banner, cli, terminal, usage


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


def test_a_middling_terminal_gets_the_logo_and_the_same_facts(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: 60)
    agent, permissions, workspace = session()

    greeting = bare(cli.greeting(agent, permissions, workspace, FakeTTY()))
    assert greeting.startswith(banner.LOGO[0])
    assert "bkht.coder" in greeting
    assert "qwen2.5-coder:14b · ask · 1,234/8,192 ctx" in greeting
    assert "/tmp/project" in greeting


# --- the box ----------------------------------------------------------------


def box(monkeypatch, width=100, **kwargs):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: width)
    agent, permissions, workspace = session()
    return bare(cli.greeting(agent, permissions, workspace, FakeTTY(), **kwargs))


def test_a_wide_terminal_gets_the_box_with_the_same_facts_in_it(monkeypatch):
    greeting = box(monkeypatch)
    assert greeting.startswith(banner.TOP_LEFT)
    assert banner.LOGO[0] in greeting
    assert "qwen2.5-coder:14b" in greeting
    assert "ask · 1,234/8,192 ctx" in greeting
    assert "/tmp/project" in greeting


def test_the_title_rides_in_the_lid_rather_than_a_row(monkeypatch):
    lid, *_ = box(monkeypatch).splitlines()
    assert usage.title() in lid


def test_every_row_of_the_box_is_the_same_width(monkeypatch):
    # Uneven rows are a box with a hole in one side, and the one place it would
    # show up first is a terminal nobody tested at.
    for width in (banner.BOX_MIN_WIDTH, 90, 200):
        rows = box(monkeypatch, width=width).splitlines()
        assert len(set(map(terminal.visible, rows))) == 1
        assert terminal.visible(rows[0]) == min(width, banner.MAX_WIDTH)


def test_a_line_too_long_for_its_column_is_cut_rather_than_wrapped(monkeypatch):
    long = "skills: " + ", ".join(f"skill-{n}" for n in range(40))
    rows = box(monkeypatch, loaded=cli.Loaded("", long)).splitlines()
    assert any("…" in row for row in rows)
    assert len(set(map(terminal.visible, rows))) == 1


def test_nothing_loaded_means_no_heading_over_the_empty_space(monkeypatch):
    assert "Loaded" not in box(monkeypatch, loaded=cli.Loaded("", ""))
    assert "Loaded" in box(monkeypatch, loaded=cli.Loaded("instructions: AGENTS.md", ""))


def test_a_skipped_skill_gets_its_own_row(monkeypatch):
    loaded = cli.Loaded("", "skills: tdd\n1 skill(s) skipped: broken.md")
    assert loaded.lines() == ["skills: tdd", "1 skill(s) skipped: broken.md"]
    greeting = box(monkeypatch, loaded=loaded)
    assert "1 skill(s) skipped: broken.md" in greeting


def test_the_box_drawn_without_colour_carries_no_escapes():
    drawn = banner.frame("coder", [None, *banner.LOGO], ["tip"], 80, colour=False)
    assert "\033" not in drawn


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
    assert bare(cli.divider()) == banner.RULE * 72


def test_a_pipe_gets_no_divider(monkeypatch):
    # A rule between exchanges is for a screen being scrolled back through; in
    # a redirected transcript it is a line of noise every turn.
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: False)
    assert cli.divider() == ""
