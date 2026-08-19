"""The shell tool.

Output is bounded and the command is timed out, because the two ways a shell
call ruins a session are an unbounded ``cat`` filling the context and a command
that never returns.

Which shell that is depends on the box: bash wherever it exists, PowerShell on
a stock Windows install, ``sh`` on a POSIX system without bash. The model only
learns the answer through the tool name and description, so both carry it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from .base import Tool, ToolError, ToolResult, Workspace, truncate

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 600

NO_SHELL = (
    "no shell is available on this system. Install Git for Windows (which "
    "provides bash), or run coder inside WSL."
)


@dataclass(frozen=True)
class Shell:
    """A resolved shell: what to call it, and how to invoke it."""

    name: str
    """The tool name the model sees."""

    label: str
    """The human name for the description."""

    argv: tuple[str, ...]
    """The argv prefix; the command is appended as the final argument."""


def resolve_shell(os_name: str = "") -> Shell | None:
    """Find the best available shell, or ``None`` if there is not one.

    Resolved once at registration so ``shutil.which`` is not paid per command.
    ``os_name`` defaults to :data:`os.name` and exists so tests can ask for the
    Windows ladder without pretending the whole process is on Windows.
    """
    os_name = os_name or os.name

    if shutil.which("bash"):
        return Shell("bash", "bash", ("bash", "-c"))

    if os_name == "nt":
        for exe in ("pwsh", "powershell"):
            if shutil.which(exe):
                return Shell(
                    "powershell", "PowerShell", (exe, "-NoProfile", "-Command")
                )
        return None

    if shutil.which("sh"):
        # Alpine and friends. The tool keeps the name `bash` so POSIX sessions,
        # transcripts, and the `/tools` listing are unchanged.
        return Shell("bash", "shell", ("sh", "-c"))

    return None


def register_shell_tools(registry, workspace: Workspace):
    """Add the shell tool to ``registry``, named for the shell that resolved."""

    shell = resolve_shell()

    def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
        if not command.strip():
            raise ToolError("command must not be empty")
        if timeout < 1 or timeout > MAX_TIMEOUT:
            raise ToolError(f"timeout must be between 1 and {MAX_TIMEOUT} seconds")
        if shell is None:
            raise ToolError(NO_SHELL)

        try:
            completed = subprocess.run(
                [*shell.argv, command],
                cwd=workspace.root,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"command timed out after {timeout}s. If it is expected to take "
                "longer, pass a larger timeout."
            ) from None
        except OSError as exc:
            raise ToolError(f"could not run {shell.argv[0]}: {exc}") from None

        parts = []
        if completed.stdout.strip():
            parts.append(truncate(completed.stdout.rstrip()))
        if completed.stderr.strip():
            parts.append("stderr:\n" + truncate(completed.stderr.rstrip()))

        output = "\n".join(parts)
        if completed.returncode != 0:
            # A non-zero exit is information, not a tool failure -- the model
            # usually needs to read the output to decide what to do next.
            return ToolResult.success(
                f"exit code {completed.returncode}\n{output}".rstrip()
            )
        return ToolResult.success(output or "(no output)")

    registry.add(
        Tool(
            name=shell.name if shell else "bash",
            description=(
                f"Run a {shell.label if shell else 'shell'} command in the "
                "workspace root. Returns stdout, stderr, and the exit code if "
                "it is non-zero."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "timeout": {"type": "integer", "description": "Seconds to wait before giving up. Defaults to 60."},
                },
                "required": ["command"],
            },
            run=bash,
            mutating=True,
        )
    )

    return registry
