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


# --- shell resolution -------------------------------------------------------


def _which(*found):
    """A ``shutil.which`` stand-in that only knows about ``found``."""
    return lambda exe: f"/usr/bin/{exe}" if exe in found else None


def test_resolve_shell_prefers_bash(monkeypatch):
    from bkht.coder.tools import shell as shell_module

    monkeypatch.setattr(shell_module.shutil, "which", _which("bash", "sh", "pwsh"))
    resolved = shell_module.resolve_shell("posix")
    assert resolved.argv == ("bash", "-c")
    assert resolved.name == "bash"


def test_resolve_shell_falls_back_to_powershell_on_nt(monkeypatch):
    from bkht.coder.tools import shell as shell_module

    monkeypatch.setattr(shell_module.shutil, "which", _which("pwsh", "powershell"))
    resolved = shell_module.resolve_shell("nt")
    assert resolved.name == "powershell"
    assert resolved.argv == ("pwsh", "-NoProfile", "-Command")

    # pwsh is preferred, but Windows PowerShell is enough on a stock box.
    monkeypatch.setattr(shell_module.shutil, "which", _which("powershell"))
    assert shell_module.resolve_shell("nt").argv == (
        "powershell",
        "-NoProfile",
        "-Command",
    )


def test_resolve_shell_falls_back_to_sh_on_posix(monkeypatch):
    from bkht.coder.tools import shell as shell_module

    monkeypatch.setattr(shell_module.shutil, "which", _which("sh"))
    resolved = shell_module.resolve_shell("posix")
    assert resolved.argv == ("sh", "-c")
    # The tool keeps its name so POSIX sessions and transcripts are unchanged.
    assert resolved.name == "bash"


def test_no_shell_is_an_actionable_tool_error(monkeypatch, project):
    from bkht.coder.tools import shell as shell_module

    monkeypatch.setattr(shell_module, "resolve_shell", lambda: None)
    registry, _ = build_registry(project)
    with pytest.raises(ToolError, match="Git for Windows"):
        bash(registry, "echo hi")
    with pytest.raises(ToolError, match="WSL"):
        bash(registry, "echo hi")


def test_description_names_the_active_shell(monkeypatch, project):
    from bkht.coder.tools import shell as shell_module

    monkeypatch.setattr(shell_module.shutil, "which", _which("pwsh"))
    windows = shell_module.resolve_shell("nt")
    monkeypatch.setattr(shell_module, "resolve_shell", lambda *a: windows)
    registry, _ = build_registry(project)

    tool = registry.get("powershell")
    assert "PowerShell" in tool.description
    assert registry.get("bash") is None
