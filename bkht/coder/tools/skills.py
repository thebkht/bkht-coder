"""The skill tool: fetch one set of standing instructions on demand.

Registered only when a workspace actually has skills. A tool the model can
never use successfully is not free -- it is one more wrong answer available at
every step -- and the tool set is kept small for exactly that reason.
"""

from __future__ import annotations

from .base import Tool, ToolError, ToolResult, contain, truncate

RESOURCE_ESCAPE = "resource '{resource}' is outside the skill's own directory"


def register_skill_tool(registry, discovery):
    """Add the ``skill`` tool, if ``discovery`` found anything to fetch."""
    if not discovery:
        return registry

    from .. import skills as skills_module

    names = ", ".join(skill.name for skill in discovery.skills)

    def skill(name: str, resource: str | None = None) -> ToolResult:
        found = discovery.get(name)
        if found is None:
            # Named rather than counted: the model picked a name from a list it
            # was shown, so the recovery is to show the list again.
            raise ToolError(f"no skill named '{name}'. Available skills: {names}")

        if resource is None:
            return ToolResult.success(skills_module.body(found))

        # A skill may ship files beside its SKILL.md, and may reach none but
        # its own -- the same boundary the workspace root has, drawn with the
        # same check.
        target = contain(found.directory, found.directory / resource)
        if target is None or not target.is_file():
            if target is None:
                raise ToolError(RESOURCE_ESCAPE.format(resource=resource))
            raise ToolError(f"skill '{name}' has no file named '{resource}'")

        from .fs import read_text

        return ToolResult.success(truncate(read_text(target)))

    registry.add(
        Tool(
            name="skill",
            description=(
                "Read the full instructions for one of the skills listed above. "
                "Call this before acting when a skill covers the task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": f"Skill name. One of: {names}."},
                    "resource": {
                        "type": "string",
                        "description": "Optional file shipped alongside the skill, by name.",
                    },
                },
                "required": ["name"],
            },
            run=skill,
        )
    )
    return registry
