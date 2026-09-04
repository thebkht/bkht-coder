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


def edit(registry, path, **kwargs):
    """`edit_file`, with the read it now requires.

    Editing a file the session has not read is refused -- see the precondition
    tests at the bottom. Every other test here is about what `edit_file` does
    once that is satisfied, so the read happens here rather than in each of
    them.
    """
    run(registry, "read_file", path=path)
    return run(registry, "edit_file", path=path, **kwargs)


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
    result = edit(
        registry, "src/util.py", old_string="x * 2", new_string="x * 3"
    )
    assert result.ok
    assert "x * 3" in (project / "src" / "util.py").read_text()


def test_edit_no_match_fails_loudly_and_says_which_case(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError) as exc:
        edit(registry, "src/util.py", old_string="nonexistent", new_string="x")
    message = str(exc.value)
    assert "was not found" in message
    assert "match the file exactly" in message


def test_edit_multi_match_fails_loudly_and_says_which_case(tools, project):
    registry, _, _ = tools
    (project / "dup.py").write_text("a = 1\nb = 1\nc = 1\n")
    with pytest.raises(ToolError) as exc:
        edit(registry, "dup.py", old_string="= 1", new_string="= 2")
    message = str(exc.value)
    assert "appears 3 times" in message
    assert "replace_all" in message
    # The two failures must be distinguishable, or the model cannot correct itself.
    assert "was not found" not in message


def test_edit_multi_match_succeeds_with_replace_all(tools, project):
    registry, _, _ = tools
    (project / "dup.py").write_text("a = 1\nb = 1\n")
    result = edit(
        registry, "dup.py", old_string="= 1", new_string="= 2", replace_all=True
    )
    assert "2 occurrences" in result.content
    assert (project / "dup.py").read_text() == "a = 2\nb = 2\n"


def test_edit_rejects_an_empty_old_string(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="must not be empty"):
        edit(registry, "src/util.py", old_string="", new_string="x")


def test_edit_rejects_a_no_op(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="identical"):
        edit(registry, "src/util.py", old_string="x * 2", new_string="x * 2")


def test_edit_on_a_missing_file(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError, match="file not found"):
        run(registry, "edit_file", path="nope.py", old_string="a", new_string="b")


def test_edit_preserves_the_rest_of_the_file(tools, project):
    registry, _, _ = tools
    before = (project / "src" / "main.py").read_text()
    edit(registry, "src/main.py", old_string="total = 0", new_string="total = 1")
    after = (project / "src" / "main.py").read_text()
    assert after == before.replace("total = 0", "total = 1")


# --- snapshots / undo -------------------------------------------------------


def test_undo_restores_an_edited_file(tools, project):
    registry, _, snapshots = tools
    before = (project / "src" / "util.py").read_text()
    edit(registry, "src/util.py", old_string="x * 2", new_string="x * 9")
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
        edit(registry, "src/util.py", old_string="nope", new_string="x")
    assert len(snapshots) == 0


# --- checking what the edit means -------------------------------------------


@pytest.fixture
def package(tmp_path):
    """A workspace where relative imports resolve, and its write tools."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "session.py").write_text("STATE_DIR = '/tmp'\n")
    (tmp_path / "pkg" / "commands.py").write_text(
        "from .session import STATE_DIR\n\n\ndef go():\n    return STATE_DIR\n"
    )
    registry, workspace = build_registry(tmp_path)
    return registry, tmp_path


def test_an_edit_importing_a_name_that_does_not_exist_still_writes(package):
    """Reported, not refused.

    Names can arrive at runtime, and a check that blocks a correct edit is
    worse than one that misses an incorrect edit.
    """
    registry, root = package
    result = edit(
        registry,
        "pkg/commands.py",
        
        old_string="from .session import STATE_DIR",
        new_string="from .session import STATE_DIR, Input",
    )
    assert result.ok
    assert "Input" in (root / "pkg" / "commands.py").read_text()


def test_the_model_is_told_about_the_name_in_the_same_breath(package):
    registry, _ = package
    result = edit(
        registry,
        "pkg/commands.py",
        
        old_string="from .session import STATE_DIR",
        new_string="from .session import STATE_DIR, Input",
    )
    assert "Warning" in result.content
    assert "session.py does not define `Input`" in result.content


def test_an_edit_that_would_not_parse_is_refused(package):
    registry, root = package
    before = (root / "pkg" / "commands.py").read_text()

    with pytest.raises(ToolError, match="unparseable"):
        edit(
            registry,
            "pkg/commands.py",
            old_string="def go():", new_string="def go(:"
        )

    assert (root / "pkg" / "commands.py").read_text() == before, "nothing was written"


def test_write_file_is_checked_the_same_way(package):
    registry, root = package
    with pytest.raises(ToolError, match="unparseable"):
        registry.get("write_file").run(path="pkg/commands.py", content="def go(:\n")
    assert "def go():" in (root / "pkg" / "commands.py").read_text()


def test_a_correct_edit_says_nothing_extra(package):
    registry, _ = package
    result = edit(
        registry,
        "pkg/commands.py",
        old_string="return STATE_DIR", new_string="return str(STATE_DIR)"
    )
    assert result.ok and "Warning" not in result.content


def test_a_non_python_file_is_written_unchecked(package):
    registry, root = package
    result = registry.get("write_file").run(path="notes.md", content="def go(:\n")
    assert result.ok and (root / "notes.md").exists()


# --- editing a file nobody read ---------------------------------------------


def test_editing_an_unread_file_is_refused(tools):
    """An exact-string match is not evidence that the model read anything.

    A string remembered from an earlier session, or reconstructed from the
    file's name, either matches or it does not. When it does not the turn
    spends three iterations finding out -- edit, `old_string was not found`,
    read, edit. When it does, the edit landed somewhere nobody looked.
    """
    registry, _, _ = tools
    with pytest.raises(ToolError, match="have not read"):
        run(registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 3")


def test_the_refusal_says_what_to_do_instead(tools):
    registry, _, _ = tools
    with pytest.raises(ToolError) as raised:
        run(registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 3")
    assert "read_file" in str(raised.value)


def test_a_file_that_was_read_may_be_edited(tools, project):
    registry, _, _ = tools
    run(registry, "read_file", path="src/util.py")
    result = run(
        registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 3"
    )
    assert result.ok
    assert "x * 3" in (project / "src" / "util.py").read_text()


def test_a_file_that_moved_since_it_was_read_is_refused(tools, project):
    """The other half of the same problem.

    A file read ten minutes ago and saved by a human since is a file whose
    remembered text is no longer what surrounds the edit -- and an exact match
    against the part that did not move is how that goes unnoticed.
    """
    registry, _, _ = tools
    run(registry, "read_file", path="src/util.py")

    target = project / "src" / "util.py"
    target.write_text("# somebody else got here first\n" + target.read_text())
    import os

    os.utime(target, (0, 0))

    with pytest.raises(ToolError, match="changed since you read it"):
        run(registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 3")


def test_a_turn_may_make_two_edits_to_one_file(tools, project):
    """Its own first write must not lock it out of the second."""
    registry, _, _ = tools
    run(registry, "read_file", path="src/util.py")
    assert run(
        registry, "edit_file", path="src/util.py", old_string="x * 2", new_string="x * 3"
    ).ok
    assert run(
        registry, "edit_file", path="src/util.py", old_string="x * 3", new_string="x * 4"
    ).ok
    assert "x * 4" in (project / "src" / "util.py").read_text()


def test_write_file_needs_no_prior_read(tools, project):
    """It states the whole file, so there is no remembered text to be wrong."""
    registry, _, _ = tools
    assert run(registry, "write_file", path="src/util.py", content="x = 1\n").ok


def test_a_written_file_may_then_be_edited(tools, project):
    registry, _, _ = tools
    run(registry, "write_file", path="fresh.py", content="value = 1\n")
    assert run(
        registry, "edit_file", path="fresh.py", old_string="value = 1", new_string="value = 2"
    ).ok
    assert (project / "fresh.py").read_text() == "value = 2\n"


def test_a_missing_file_keeps_its_own_error(tools):
    """"You have not read it" is true of a file that is not there, and useless."""
    registry, _, _ = tools
    with pytest.raises(ToolError, match="file not found"):
        run(registry, "edit_file", path="nope.py", old_string="a", new_string="b")
