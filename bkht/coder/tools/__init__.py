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
    root: Path | str,
    read_only: bool = False,
    snapshots=None,
    skills=None,
    jobs=None,
    session=None,
    delegate=None,
    agent_tools=None,
) -> tuple[Registry, Workspace]:
    """Build the tool set for a workspace rooted at ``root``.

    ``read_only`` omits the mutating tools entirely rather than denying them at
    call time, so the model is never tempted by a tool it cannot use. ``skills``
    is omitted for the same reason when a workspace has none: no skills, no
    ``skill`` tool, and a tool set byte-for-byte what it was before skills
    existed.

    ``agent_tools`` is what a workspace wrote for itself, already discovered
    and already permitted -- see :mod:`~bkht.coder.usertools` for the three
    things that have to be true before there is anything in it.

    ``session`` and ``delegate`` follow that same rule, and are why a sub-agent
    gets neither ``plan`` nor ``task``: the first is offered only to a loop that
    has a session to write a plan onto, the second only when a provider is
    handed in to run it with. A nested registry is built with neither, so a
    delegated task cannot delegate, and cannot rewrite the plan of the turn that
    delegated to it.
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

    # Registered before the mutating tools, so it is offered in plan mode too:
    # producing a plan is the whole of what plan mode is for.
    if session is not None:
        from .plan import register_plan_tool

        register_plan_tool(registry, session)

    if delegate is not None:
        from .task import register_task_tool

        register_task_tool(registry, root, **delegate)

    if not read_only:
        from .fs import register_write_tools
        from .shell import register_shell_tools

        register_write_tools(registry, workspace, snapshots)
        register_shell_tools(registry, workspace)

        if jobs is not None:
            from .background import register_background_tools

            register_background_tools(registry, workspace, jobs)

    # Last, so that every built-in name already exists to collide with: a
    # user tool may add to this set and may not quietly take a name over.
    if agent_tools:
        from .. import usertools

        agent_tools.problems.extend(
            usertools.register(registry, agent_tools, read_only=read_only)
        )

    return registry, workspace
