"""read_file and list_files."""

import pytest

from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


@pytest.fixture
def tools(project):
    registry, workspace = build_registry(project, read_only=True)
    return registry, workspace


def run(registry, name, **kwargs):
    return registry.get(name).run(**kwargs)


def test_read_file_returns_numbered_lines(tools):
    registry, _ = tools
    out = run(registry, "read_file", path="src/util.py").content
    assert out.splitlines()[0] == "1\tdef helper(x):"


def test_read_file_paging_reports_the_window(tools):
    registry, _ = tools
    out = run(registry, "read_file", path="src/main.py", offset=2, limit=2).content
    assert out.splitlines()[0].startswith("2\t")
    assert "[showing lines 2-3 of 5]" in out


def test_read_file_full_read_has_no_window_note(tools):
    registry, _ = tools
    assert "[showing lines" not in run(registry, "read_file", path="src/main.py").content


def test_read_file_missing_file(tools):
    registry, _ = tools
    with pytest.raises(ToolError, match="file not found"):
        run(registry, "read_file", path="nope.py")


def test_read_file_on_a_directory(tools):
    registry, _ = tools
    with pytest.raises(ToolError, match="is a directory"):
        run(registry, "read_file", path="src")


def test_read_file_rejects_binary(tools, project):
    registry, _ = tools
    (project / "blob.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        run(registry, "read_file", path="blob.bin")


def test_read_file_offset_past_end(tools):
    registry, _ = tools
    with pytest.raises(ToolError, match="past the end"):
        run(registry, "read_file", path="src/util.py", offset=99)


def test_read_file_rejects_zero_offset(tools):
    registry, _ = tools
    with pytest.raises(ToolError, match="1-based"):
        run(registry, "read_file", path="src/util.py", offset=0)


def test_read_file_empty_file(tools, project):
    registry, _ = tools
    (project / "empty.py").write_text("")
    assert "is empty" in run(registry, "read_file", path="empty.py").content


def test_read_file_escape_is_rejected(tools):
    registry, _ = tools
    with pytest.raises(ToolError, match="outside the workspace"):
        run(registry, "read_file", path="../../etc/passwd")


def test_list_files_skips_noise(tools):
    registry, _ = tools
    out = run(registry, "list_files").content
    assert "main.py" in out and "README.md" in out
    assert "node_modules" not in out and ".git" not in out


def test_list_files_respects_depth(tools):
    registry, _ = tools
    shallow = run(registry, "list_files", depth=1).content
    assert "src/" in shallow and "main.py" not in shallow


def test_list_files_on_a_file(tools):
    registry, _ = tools
    with pytest.raises(ToolError, match="not a directory"):
        run(registry, "list_files", path="README.md")


def test_read_only_registry_has_no_mutating_tools(tools):
    registry, _ = tools
    assert registry.names() == ["list_files", "read_file"]
