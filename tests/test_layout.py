"""The `agent/` surface: which roots are adopted, and what they hold."""

from pathlib import Path

import pytest

from bkht.coder import layout
from bkht.coder.layout import Surface, inventory, render, surface


@pytest.fixture(autouse=True)
def no_global(monkeypatch, tmp_path):
    """Point the global root somewhere empty.

    Every test here would otherwise pick up whatever the developer running the
    suite happens to have in ``~/.bkht-coder/agent``.
    """
    monkeypatch.setattr(layout, "GLOBAL_ROOT", tmp_path / "nowhere" / "agent")


def mark(root: Path, contents: str = "{}") -> Path:
    directory = root / "agent"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.json").write_text(contents)
    return directory


# --- adoption -------------------------------------------------------------


def test_a_marked_directory_is_adopted(tmp_path):
    marked = mark(tmp_path)
    assert [root.path for root in surface(tmp_path).roots] == [marked]


def test_no_agent_directory_is_no_surface(tmp_path):
    found = surface(tmp_path)
    assert not found and not found.problems


def test_an_unmarked_agent_directory_is_passed_over_in_silence(tmp_path):
    # The eve case: this directory is that project's agent, not ours. Ignoring
    # it is the point, and complaining about it would be complaining about
    # somebody else's file.
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "instructions.md").write_text("You are a data agent.\n")

    found = surface(tmp_path)
    assert not found.roots and not found.problems


def test_a_broken_marker_is_reported(tmp_path):
    # This one the user wrote meaning it to work, so silence would be wrong.
    mark(tmp_path, "{not json")
    found = surface(tmp_path)
    assert not found.roots
    assert "agent.json" in found.problems[0]


def test_a_marker_that_is_not_an_object_is_reported(tmp_path):
    mark(tmp_path, "[]")
    found = surface(tmp_path)
    assert not found.roots and found.problems


def test_the_global_root_needs_no_marker(tmp_path, monkeypatch):
    global_root = tmp_path / "home" / "agent"
    (global_root / "skills").mkdir(parents=True)
    monkeypatch.setattr(layout, "GLOBAL_ROOT", global_root)

    assert [root.path for root in surface(tmp_path).roots] == [global_root]


def test_the_global_root_can_be_left_out(tmp_path, monkeypatch):
    global_root = tmp_path / "home" / "agent"
    global_root.mkdir(parents=True)
    monkeypatch.setattr(layout, "GLOBAL_ROOT", global_root)

    assert surface(tmp_path, include_global=False).roots == []


def test_the_workspace_root_comes_last(tmp_path, monkeypatch):
    # Least specific first, so a caller layering its own discovery over these
    # gets the workspace winning by walking them in order.
    global_root = tmp_path / "home" / "agent"
    global_root.mkdir(parents=True)
    monkeypatch.setattr(layout, "GLOBAL_ROOT", global_root)
    marked = mark(tmp_path)

    assert [(root.path, root.scope) for root in surface(tmp_path).roots] == [
        (global_root, layout.GLOBAL), (marked, layout.WORKSPACE),
    ]


# --- slots ----------------------------------------------------------------


def test_a_slot_lists_only_directories_that_exist(tmp_path):
    marked = mark(tmp_path)
    (marked / "skills").mkdir()

    found = surface(tmp_path)
    assert found.slot("skills") == [marked / "skills"]
    assert found.slot("commands") == []


def test_a_slot_that_is_a_file_is_not_a_slot(tmp_path):
    marked = mark(tmp_path)
    (marked / "skills").write_text("not a directory\n")

    assert surface(tmp_path).slot("skills") == []


def test_instructions_may_be_one_file(tmp_path):
    marked = mark(tmp_path)
    (marked / "instructions.md").write_text("Rules.\n")

    assert layout.instructions(marked) == [marked / "instructions.md"]


def test_instructions_may_be_a_directory_composed_in_order(tmp_path):
    marked = mark(tmp_path)
    (marked / "instructions").mkdir()
    for name in ("20-style.md", "10-house.md"):
        (marked / "instructions" / name).write_text(name)

    assert [path.name for path in layout.instructions(marked)] == ["10-house.md", "20-style.md"]


def test_a_directory_of_instructions_wins_over_a_leftover_file(tmp_path):
    # Both is the shape of a half-finished migration, and the file is the half
    # that was left behind.
    marked = mark(tmp_path)
    (marked / "instructions.md").write_text("old\n")
    (marked / "instructions").mkdir()
    (marked / "instructions" / "new.md").write_text("new\n")

    assert layout.instructions(marked) == [marked / "instructions" / "new.md"]


def test_hooks_are_listed_one_level_down(tmp_path):
    marked = mark(tmp_path)
    (marked / "hooks" / "post_tool").mkdir(parents=True)
    (marked / "hooks" / "post_tool" / "format.sh").write_text("#!/bin/sh\n")

    listed = layout.entries(marked / "hooks", "hooks")
    assert [path.name for path in listed] == ["format.sh"]


# --- printing -------------------------------------------------------------


def test_the_inventory_names_every_slot_that_has_something_in_it(tmp_path):
    marked = mark(tmp_path)
    (marked / "instructions.md").write_text("Rules.\n")
    (marked / "commands").mkdir()
    (marked / "commands" / "review.md").write_text("Review it.\n")

    [(name, slots)] = inventory(surface(tmp_path), tmp_path)
    assert name == "agent"
    assert [slot for slot, _ in slots] == ["instructions", "commands"]
    assert slots[1][1] == ["commands/review.md"]


def test_nothing_found_says_where_to_put_it(tmp_path):
    printed = render(surface(tmp_path), tmp_path)
    assert "agent.json" in printed


def test_problems_are_printed_under_the_listing(tmp_path):
    printed = render(Surface(problems=["agent/agent.json: broken"]), tmp_path)
    assert "problem: agent/agent.json: broken" in printed


def test_an_empty_slot_directory_is_still_listed(tmp_path):
    # Made and not yet filled is part of what the user meant. Hiding it would
    # read as though the directory had been spelled wrong.
    marked = mark(tmp_path)
    (marked / "hooks").mkdir()

    [(_, slots)] = inventory(surface(tmp_path), tmp_path)
    assert slots == [("hooks", [])]
