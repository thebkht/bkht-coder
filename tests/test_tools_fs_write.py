"""write_file, edit_file, and the snapshots that back undo."""

import pytest

from bkht.coder.session import Snapshots
from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


@pytest.fixture
def tools(project):
    snapshots = Snapshots()
    registry, workspace = build_registry(project, snapshots=snapshots)
    return registry, workspace, snapshots


def run(registry, name, **kwargs):
    return registry.get(name).run(**kwargs)


# --- write_file -------------------------------------------------------------


def test_write_creates_a_file(tools, project):
    registry, _, _ = tools
    result = run(registry, "write_file", path="new.py", content="x = 1\n")
    assert result.ok and "Created new.py" in result.content
    assert (project / "new.py").read_text() == "x = 1\n"


def test_write_creates_missing_parents(tools, project):
    registry, _, _ = tools
    run(registry, "write_file", path="a/b/c.py", content="pass\n")
    assert (project / "a" / "b" / "c.py").exists()


def test_write_replaces_and_says_so(tools, project):
    registry, _, _ = tools
    result = run(registry, "write_file", path="README.md", content="# gone\n")
    assert "Updated README.md" in result.content
    assert (project / "README.md").read_text() == "# gone\n"


def test_write_refuses_a_directory(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="is a directory"):
        run(registry, "write_file", path="src", content="x")


def test_write_cannot_escape_the_workspace(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="outside the workspace"):
        run(registry, "write_file", path="../escaped.py", content="x")


# --- edit_file --------------------------------------------------------------


def test_edit_replaces_a_unique_string(tools, project):
    registry, _, _ = tools
    result = run(
        registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 3"
    )
    assert result.ok
    assert "x * 3" in (project / "src" / "util.py").read_text()


def test_edit_no_match_fails_loudly_and_says_which_case(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError) as exc:
        run(registry, "edit_file", path="src/util.py", old_string="nonexistent", new_string="x")
    message = str(exc.value)
    assert "was not found" in message
    assert "match the file exactly" in message


def test_edit_multi_match_fails_loudly_and_says_which_case(tools, project):
    registry, _, _ = tools
    (project / "dup.py").write_text("a = 1\nb = 1\nc = 1\n")
    with pytest.raises(ToolError) as exc:
        run(registry, "edit_file", path="dup.py", old_string="= 1", new_string="= 2")
    message = str(exc.value)
    assert "appears 3 times" in message
    assert "replace_all" in message
    # The two failures must be distinguishable, or the model cannot correct itself.
    assert "was not found" not in message


def test_edit_multi_match_succeeds_with_replace_all(tools, project):
    registry, _, _ = tools
    (project / "dup.py").write_text("a = 1\nb = 1\n")
    result = run(
        registry, "edit_file", path="dup.py", old_string="= 1", new_string="= 2", replace_all=True
    )
    assert "2 occurrences" in result.content
    assert (project / "dup.py").read_text() == "a = 2\nb = 2\n"


def test_edit_rejects_an_empty_old_string(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="must not be empty"):
        run(registry, "edit_file", path="src/util.py", old_string="", new_string="x")


def test_edit_rejects_a_no_op(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="identical"):
        run(registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 2")


def test_edit_on_a_missing_file(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="file not found"):
        run(registry, "edit_file", path="nope.py", old_string="a", new_string="b")


def test_edit_preserves_the_rest_of_the_file(tools, project):
    registry, _, _ = tools
    before = (project / "src" / "main.py").read_text()
    run(registry, "edit_file", path="src/main.py", old_string="total = 0", new_string="total = 1")
    after = (project / "src" / "main.py").read_text()
    assert after == before.replace("total = 0", "total = 1")


# --- snapshots / undo -------------------------------------------------------


def test_undo_restores_an_edited_file(tools, project):
    registry, _, snapshots = tools
    before = (project / "src" / "util.py").read_text()
    run(registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 9")
    assert snapshots.undo() == "restored util.py"
    assert (project / "src" / "util.py").read_text() == before


def test_undo_deletes_a_created_file(tools, project):
    registry, _, snapshots = tools
    run(registry, "write_file", path="new.py", content="x = 1\n")
    assert snapshots.undo() == "removed new.py"
    assert not (project / "new.py").exists()


def test_undo_unwinds_one_step_at_a_time(tools, project):
    registry, _, snapshots = tools
    run(registry, "write_file", path="a.py", content="1\n")
    run(registry, "write_file", path="a.py", content="2\n")
    snapshots.undo()
    assert (project / "a.py").read_text() == "1\n"
    snapshots.undo()
    assert not (project / "a.py").exists()


def test_undo_on_an_empty_stack(tools):
    _, _, snapshots = tools
    assert snapshots.undo() is None


def test_a_failed_edit_leaves_no_snapshot(tools):
    registry, _, snapshots = tools
    with pytest.raises(ToolError):
        run(registry, "edit_file", path="src/util.py", old_string="nope", new_string="x")
    assert len(snapshots) == 0
