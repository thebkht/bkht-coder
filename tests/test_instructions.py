"""Instruction files: discovery, precedence, and the context budget."""

from pathlib import Path

import pytest

from bkht.coder import instructions as I
from bkht.coder.instructions import (
    Instruction,
    load_instructions,
    render,
    summarize,
)


@pytest.fixture
def no_global(monkeypatch, tmp_path):
    """Point the global source at an empty directory.

    Without this the developer's own ~/.bkht-coder/AGENTS.md would leak into
    the tests and make them pass or fail depending on whose machine they run on.
    """
    monkeypatch.setattr(I, "GLOBAL_DIR", tmp_path / "nowhere")


def test_no_files_found_is_empty(tmp_path, no_global):
    assert load_instructions(tmp_path) == []
    assert render([]) == ""
    assert summarize([]) == ""


def test_reads_agents_md(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("Use tabs.\n")
    loaded = load_instructions(tmp_path)
    assert [i.text for i in loaded] == ["Use tabs."]
    assert loaded[0].source.endswith("AGENTS.md")


def test_reads_claude_md(tmp_path, no_global):
    (tmp_path / "CLAUDE.md").write_text("Never touch generated files.\n")
    assert [i.text for i in load_instructions(tmp_path)] == [
        "Never touch generated files."
    ]


def test_both_workspace_files_load_in_order(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("first")
    (tmp_path / "CLAUDE.md").write_text("second")
    assert [i.text for i in load_instructions(tmp_path)] == ["first", "second"]


def test_global_comes_before_workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("global rule")
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text("project rule")

    # Least specific first: the workspace's own rules are read last, so they
    # are what the model sees most recently.
    assert [i.text for i in load_instructions(root)] == ["global rule", "project rule"]


def test_global_can_be_disabled(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("global rule")
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    assert load_instructions(root, include_global=False) == []


def test_empty_file_is_not_a_source(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("   \n\n  ")
    assert load_instructions(tmp_path) == []


def test_binary_file_is_skipped_not_fatal(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_bytes(b"rules\x00binary")
    (tmp_path / "CLAUDE.md").write_text("readable")
    # The unreadable source drops out; the agent still starts, with the rules
    # it could read.
    assert [i.text for i in load_instructions(tmp_path)] == ["readable"]


def test_directory_named_like_an_instruction_file_is_skipped(tmp_path, no_global):
    (tmp_path / "AGENTS.md").mkdir()
    assert load_instructions(tmp_path) == []


def test_per_source_truncation_is_announced(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("x" * 5000)
    loaded = load_instructions(tmp_path, per_source=100)
    assert loaded[0].truncated is True
    assert I.TRUNCATION_NOTE in loaded[0].text
    assert len(loaded[0].text) < 5000


def test_total_budget_is_spent_from_the_most_specific_source_back(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("g" * 200)
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text("p" * 200)

    # Only the workspace file fits. A long global file must not crowd out the
    # rules that belong to the project actually being worked on.
    loaded = load_instructions(root, budget=200, per_source=1000)
    assert [i.text for i in loaded] == ["p" * 200]


def test_a_source_that_partly_fits_is_kept_and_marked(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("g" * 200)
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text("p" * 100)

    loaded = load_instructions(root, budget=150, per_source=1000)
    assert len(loaded) == 2
    assert loaded[1].text == "p" * 100
    assert loaded[0].truncated is True


def test_home_relative_sources_are_labelled_with_a_tilde(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("global rule")
    monkeypatch.setattr(I, "GLOBAL_DIR", home)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    loaded = load_instructions(tmp_path / "project")
    assert loaded[0].source == "~/home/AGENTS.md"


def test_render_names_every_source(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("alpha")
    (tmp_path / "CLAUDE.md").write_text("beta")
    text = render(load_instructions(tmp_path))
    assert "alpha" in text and "beta" in text
    assert text.count("From ") == 2


def test_summarize_marks_truncation():
    assert summarize([Instruction("a.md", "x")]) == "instructions: a.md"
    assert "truncated" in summarize([Instruction("a.md", "x", truncated=True)])
