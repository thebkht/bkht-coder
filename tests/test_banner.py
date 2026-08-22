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


def test_a_terminal_gets_the_logo_and_the_same_facts(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: 100)
    agent, permissions, workspace = session()

    greeting = cli.greeting(agent, permissions, workspace, FakeTTY())
    assert greeting.startswith(banner.LOGO[0])
    assert "bkht.coder" in greeting
    assert "qwen2.5-coder:14b · ask · 1,234/8,192 ctx" in greeting
    assert "/tmp/project" in greeting


@pytest.mark.parametrize("width", [40, banner.MIN_WIDTH - 1])
def test_a_narrow_terminal_falls_back_rather_than_wrapping(monkeypatch, width):
    monkeypatch.setattr(terminal, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "width", lambda *a, **k: width)
    agent, permissions, workspace = session()

    greeting = cli.greeting(agent, permissions, workspace, FakeTTY())
    assert "⣿" not in greeting
    assert greeting.startswith("coder · ")
