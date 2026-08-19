"""Diff collection and chunking. No model needed."""

import subprocess

import pytest

from bkht.coder.review.diff import (
    GitError,
    chunk,
    collect_diff,
    collect_files,
    parse_diff,
)


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "user.email=t@localhost", "-c", "user.name=t", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    """A repo on branch main with one committed file."""
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def divide(a, b):\n"
        "    return a / b\n"
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


# --- line numbers -----------------------------------------------------------

SAMPLE = """\
diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,6 +1,7 @@ def add(a, b):
 def add(a, b):
-    return a + b
+    return a + b + 0
+
 
 
 def divide(a, b):
"""


def test_added_lines_get_new_file_numbers():
    [file] = parse_diff(SAMPLE)
    added = [c for c in file.hunks[0].changes if c.kind == "+"]
    assert [(c.line, c.text) for c in added] == [(2, "    return a + b + 0"), (3, "")]


def test_removed_lines_have_no_new_file_number():
    [file] = parse_diff(SAMPLE)
    removed = [c for c in file.hunks[0].changes if c.kind == "-"]
    assert removed[0].line is None


def test_context_lines_are_numbered_too():
    # A problem is often on a context line, so it needs a citable number.
    [file] = parse_diff(SAMPLE)
    context = [c for c in file.hunks[0].changes if c.kind == " "]
    assert context[0].line == 1
    assert context[-1].line == 6


def test_hunk_header_context_is_kept():
    [file] = parse_diff(SAMPLE)
    assert file.hunks[0].context == "def add(a, b):"


def test_render_shows_numbers_and_markers():
    [file] = parse_diff(SAMPLE)
    rendered = file.hunks[0].render()
    assert "    2 +    return a + b + 0" in rendered
    assert "    - -    return a + b" in rendered


# --- collection -------------------------------------------------------------


def test_uncommitted_covers_staged_and_unstaged(repo):
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    git(repo, "add", "calc.py")
    (repo / "new.py").write_text("x = 1\n")
    git(repo, "add", "new.py")
    (repo / "calc.py").write_text("def add(a, b):\n    return a * b\n")

    files = {f.path: f for f in collect_diff(repo)}
    assert set(files) == {"calc.py", "new.py"}
    assert files["new.py"].status == "added"


def test_staged_only(repo):
    (repo / "staged.py").write_text("x = 1\n")
    git(repo, "add", "staged.py")
    (repo / "unstaged.py").write_text("y = 2\n")

    assert [f.path for f in collect_diff(repo, staged=True)] == ["staged.py"]


def test_base_uses_the_merge_base_not_the_branch_tip(repo):
    git(repo, "checkout", "-qb", "feature")
    (repo / "feature.py").write_text("f = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "feature work")

    # A commit lands on main after the branch point. It must not be attributed
    # to this branch, which is exactly what diffing against the tip would do.
    git(repo, "checkout", "-q", "main")
    (repo / "unrelated.py").write_text("u = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "unrelated")
    git(repo, "checkout", "-q", "feature")

    assert [f.path for f in collect_diff(repo, base="main")] == ["feature.py"]


def test_explicit_revision_range(repo):
    (repo / "second.py").write_text("s = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "second")
    assert [f.path for f in collect_diff(repo, revision_range="HEAD~1..HEAD")] == [
        "second.py"
    ]


def test_a_clean_tree_yields_nothing(repo):
    assert collect_diff(repo) == []


def test_deletions_are_not_reviewed(repo):
    (repo / "calc.py").unlink()
    assert collect_diff(repo) == []


def test_not_a_repository_is_a_clear_error(tmp_path):
    with pytest.raises(GitError, match="not a git repository"):
        collect_diff(tmp_path)


def test_unknown_base_is_a_clear_error(repo):
    with pytest.raises(GitError):
        collect_diff(repo, base="no-such-branch")


def test_whole_files_read_as_all_additions(repo):
    [file] = collect_files(repo, ["calc.py"])
    assert file.status == "whole file"
    assert all(c.kind == "+" for c in file.hunks[0].changes)
    assert file.hunks[0].changes[0].line == 1


def test_whole_files_rejects_a_missing_path(repo):
    with pytest.raises(GitError, match="not a file"):
        collect_files(repo, ["nope.py"])


# --- chunking ---------------------------------------------------------------


def test_a_small_diff_is_one_unit():
    units = chunk(parse_diff(SAMPLE))
    assert len(units) == 1
    assert units[0].paths == ["calc.py"]


def test_a_large_diff_becomes_several_units():
    files = parse_diff(SAMPLE)
    files = files * 20
    units = chunk(files, budget=1000)
    assert len(units) > 1


def test_no_hunk_is_ever_split(repo):
    # Half a function is worse than none: a model shown half will confidently
    # review the part it cannot see.
    files = parse_diff(SAMPLE) * 10
    total_hunks = sum(len(f.hunks) for f in files)
    units = chunk(files, budget=200)
    assert sum(len(f.hunks) for u in units for f in u.files) == total_hunks


def test_an_oversized_hunk_gets_its_own_unit_rather_than_truncation():
    big = SAMPLE.replace(
        "+    return a + b + 0", "\n".join(f"+    x{i} = {i}" for i in range(400))
    )
    units = chunk(parse_diff(big), budget=500)
    assert len(units) == 1
    assert units[0].size() > 500


def test_hunks_from_one_file_stay_together_in_a_unit():
    two_hunks = SAMPLE + """\
@@ -20,3 +21,4 @@ def divide(a, b):
 def divide(a, b):
-    return a / b
+    return a // b
"""
    [unit] = chunk(parse_diff(two_hunks))
    assert len(unit.files) == 1
    assert len(unit.files[0].hunks) == 2


def test_an_empty_diff_yields_no_units():
    assert chunk([]) == []
