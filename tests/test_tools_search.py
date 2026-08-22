"""grep and glob."""

import pytest

from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


@pytest.fixture
def registry(project):
    registry, _ = build_registry(project, read_only=True)
    return registry


def run(registry, name, **kwargs):
    return registry.get(name).run(**kwargs).content


def test_grep_reports_file_and_line(registry):
    out = run(registry, "grep", pattern=r"def helper")
    assert out == "src/util.py:1: def helper(x):"


def test_grep_is_a_regex(registry):
    assert "src/main.py:1" in run(registry, "grep", pattern=r"^def \w+\(\):")


def test_grep_glob_narrows_the_scan(registry):
    assert "No matches" in run(registry, "grep", pattern="demo", glob="*.py")
    assert "README.md:1" in run(registry, "grep", pattern="demo", glob="*.md")


def test_grep_skips_ignored_directories(registry):
    assert "node_modules" not in run(registry, "grep", pattern="module")


def test_grep_no_match_is_a_success_with_a_clear_message(registry):
    result = registry.get("grep").run(pattern="zzzz")
    assert result.ok and "No matches" in result.content


def test_grep_invalid_regex_is_an_actionable_error(registry):
    with pytest.raises(ToolError, match="invalid regular expression"):
        run(registry, "grep", pattern="(unclosed")


def test_grep_skips_binaries_rather_than_failing(registry, project):
    (project / "blob.bin").write_bytes(b"\x00\xff" * 100)
    assert "No matches" in run(registry, "grep", pattern="zzzz")


def test_glob_finds_by_bare_extension(registry):
    # A small model writes *.py as often as **/*.py; both must work.
    assert sorted(run(registry, "glob", pattern="*.py").splitlines()) == [
        "src/main.py",
        "src/util.py",
    ]


def test_glob_finds_by_recursive_pattern(registry):
    assert "src/main.py" in run(registry, "glob", pattern="**/*.py")


def test_glob_no_match_is_a_clear_message(registry):
    assert "No files match" in run(registry, "glob", pattern="*.rs")


def test_read_only_registry_tool_set(registry):
    assert registry.names() == ["glob", "grep", "list_files", "read_file"]


# --- the walk itself ------------------------------------------------------


def test_the_walk_stops_when_the_caller_does(project, monkeypatch):
    # Every cap in the search and scout paths bounds how many files they read,
    # which only bounds their cost if the walk itself can be abandoned. Sorting
    # the whole tree up front meant the work was already done by the time the
    # first result appeared.
    from pathlib import Path

    from bkht.coder.tools.search import iter_files

    for index in range(50):
        package = project / f"pkg{index:02d}"
        package.mkdir()
        (package / "mod.py").write_text("value = 1\n")

    visited = []
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: visited.append(self) or original(self))

    next(iter_files(project, project))
    assert len(visited) < 10, "the whole tree was walked to produce one file"


def test_an_ignored_directory_is_never_entered(project, monkeypatch):
    # node_modules is not merely filtered out of the results; it is not read.
    from pathlib import Path

    from bkht.coder.tools.search import iter_files

    visited = []
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: visited.append(self) or original(self))

    list(iter_files(project, project))
    assert not any(path.name == "node_modules" for path in visited)


def test_a_directory_symlinked_to_its_own_parent_terminates(project):
    # Rare in a repository, unavoidable in a home directory, and fatal either
    # way: the walk would never end.
    from bkht.coder.tools.search import iter_files

    (project / "loop").symlink_to(project)
    assert any(path.name == "main.py" for path in iter_files(project, project))
