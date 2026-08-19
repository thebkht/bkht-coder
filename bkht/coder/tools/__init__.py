"""Tool registry construction.

The set is deliberately small. Every extra tool measurably degrades selection
accuracy on a 14b model, so tools are added only when they earn their place.
"""

from __future__ import annotations

from pathlib import Path

from .base import Registry, Tool, ToolError, ToolResult, Workspace, validate_arguments
from .fs import register_read_tools

__all__ = [
    "Registry",
    "Tool",
    "ToolError",
    "ToolResult",
    "Workspace",
    "build_registry",
    "validate_arguments",
]


def build_registry(root: Path | str, read_only: bool = False) -> tuple[Registry, Workspace]:
    """Build the tool set for a workspace rooted at ``root``.

    ``read_only`` omits the mutating tools entirely rather than denying them at
    call time, so the model is never tempted by a tool it cannot use.
    """
    workspace = Workspace(Path(root))
    registry = Registry()
    register_read_tools(registry, workspace)

    if not read_only:
        from .fs import register_write_tools
        from .shell import register_shell_tools

        register_write_tools(registry, workspace)
        register_shell_tools(registry, workspace)

    return registry, workspace
