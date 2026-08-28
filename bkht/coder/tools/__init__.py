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


def build_registry(
    root: Path | str, read_only: bool = False, snapshots=None, skills=None, jobs=None
) -> tuple[Registry, Workspace]:
    """Build the tool set for a workspace rooted at ``root``.

    ``read_only`` omits the mutating tools entirely rather than denying them at
    call time, so the model is never tempted by a tool it cannot use. ``skills``
    is omitted for the same reason when a workspace has none: no skills, no
    ``skill`` tool, and a tool set byte-for-byte what it was before skills
    existed.
    """
    workspace = Workspace(Path(root))
    registry = Registry()
    register_read_tools(registry, workspace)

    from .search import register_search_tools

    register_search_tools(registry, workspace)

    # Registered because the agent's opening scout already shows the model a
    # `codebase_search` result; a tool it has seen the output of has to exist.
    from ..retrieval import register_retrieval_tool

    register_retrieval_tool(registry, workspace)

    # Reading a CI log is not a mutation, so these are offered in plan mode too.
    # Each registers only when its CLI is installed: a tool the model can see
    # and cannot use is a turn spent finding that out.
    from .github import register_github_tool
    from .gitlab import register_gitlab_tool

    register_github_tool(registry)
    register_gitlab_tool(registry)

    if skills:
        from .skills import register_skill_tool

        register_skill_tool(registry, skills)

    if not read_only:
        from .fs import register_write_tools
        from .shell import register_shell_tools

        register_write_tools(registry, workspace, snapshots)
        register_shell_tools(registry, workspace)

        if jobs is not None:
            from .background import register_background_tools

            register_background_tools(registry, workspace, jobs)

    return registry, workspace
