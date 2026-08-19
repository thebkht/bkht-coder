"""The bash tool.

Output is bounded and the command is timed out, because the two ways a shell
call ruins a session are an unbounded ``cat`` filling the context and a command
that never returns.
"""

from __future__ import annotations

import subprocess

from .base import Tool, ToolError, ToolResult, Workspace, truncate

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 600


def register_shell_tools(registry, workspace: Workspace):
    """Add ``bash`` to ``registry``."""

    def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
        if not command.strip():
            raise ToolError("command must not be empty")
        if timeout < 1 or timeout > MAX_TIMEOUT:
            raise ToolError(f"timeout must be between 1 and {MAX_TIMEOUT} seconds")

        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=workspace.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"command timed out after {timeout}s. If it is expected to take "
                "longer, pass a larger timeout."
            ) from None
        except OSError as exc:
            raise ToolError(f"could not run command: {exc}") from None

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
            name="bash",
            description=(
                "Run a shell command in the workspace root. Returns stdout, "
                "stderr, and the exit code if it is non-zero."
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
