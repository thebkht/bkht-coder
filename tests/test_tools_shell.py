"""The bash tool."""

import pytest

from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


@pytest.fixture
def registry(project):
    registry, _ = build_registry(project)
    return registry


def bash(registry, command, **kwargs):
    return registry.get("bash").run(command=command, **kwargs)


def test_stdout_is_returned(registry):
    assert bash(registry, "echo hello").content == "hello"


def test_runs_in_the_workspace_root(registry, project):
    assert bash(registry, "pwd").content == str(project.resolve())


def test_non_zero_exit_is_information_not_a_failure(registry):
    result = bash(registry, "echo oops >&2; exit 3")
    assert result.ok
    assert "exit code 3" in result.content
    assert "oops" in result.content


def test_no_output_is_stated_explicitly(registry):
    assert bash(registry, "true").content == "(no output)"


def test_timeout_is_actionable(registry):
    with pytest.raises(ToolError, match="timed out after 1s"):
        bash(registry, "sleep 5", timeout=1)


def test_timeout_bounds_are_enforced(registry):
    with pytest.raises(ToolError, match="between 1 and 600"):
        bash(registry, "true", timeout=0)


def test_empty_command_is_rejected(registry):
    with pytest.raises(ToolError, match="must not be empty"):
        bash(registry, "   ")


def test_huge_output_is_truncated_with_a_marker(registry):
    result = bash(registry, "seq 1 100000")
    assert "[truncated" in result.content
    assert len(result.content.splitlines()) < 500
