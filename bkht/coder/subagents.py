"""Specialist sub-agents, written as directories.

The ``task`` tool already hands a question to a second agent: it reads in a
window of its own and gives back prose, so the parent pays for the answer and
not for the search. What it could not do was hand that question to somebody in
particular. Every delegated task got the same empty prompt and the same skills,
which is right for "find where the registry is built" and wrong for "review
this diff the way we review diffs here".

A subagent is a directory saying who to ask:

    agent/subagents/reviewer/
    ├── agent.md          description: what this one is for
    ├── instructions.md   its standing rules, in place of the parent's
    └── skills/           its own, not the workspace's

``agent.md`` needs a ``description`` and nothing else. That description is the
only thing the model sees when it picks one, so it is written for choosing
between them -- what this agent is for, not how it works.

Deliberately not here: a per-subagent model. Naming a second model on a machine
serving one means evicting the first to load it and evicting it back
afterwards, which costs more than the delegation saves. A subagent runs on the
session's model, like every other part of the turn.

The three bounds ``task`` documents are unchanged, and they are why this is
safe to add: a subagent is read-only, cannot delegate further, and gets a
share of a turn rather than a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import layout
from .skills import Discovery, parse_frontmatter, scan
from .tools.base import ToolError
from .tools.fs import read_text

#: What makes a directory a subagent rather than a directory.
FILE = "agent.md"
INSTRUCTIONS = "instructions.md"
SKILLS = "skills"

MAX_NAME = 64
MAX_DESCRIPTION = 200
#: One subagent's instructions, in the prompt of the agent that runs it. Larger
#: than a skill's listing and smaller than the workspace's own budget: this is
#: a specialist's brief, not a project's rulebook.
MAX_INSTRUCTIONS = 2_000


@dataclass(frozen=True)
class Subagent:
    """One specialist: who it is, what it is told, and what it may look up."""

    name: str
    description: str
    path: Path
    source: str
    instructions: str = ""
    skills: Discovery = field(default_factory=Discovery)


@dataclass
class Found:
    """What a scan found, including what it refused.

    Problems are carried rather than raised, for the reason
    :class:`~bkht.coder.skills.Discovery` carries them: a subagent that never
    loads looks exactly like one the model chose not to call.
    """

    agents: list[Subagent] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def get(self, name: str) -> Subagent | None:
        wanted = (name or "").strip().lower()
        return next((a for a in self.agents if a.name == wanted), None)

    def names(self) -> list[str]:
        return [agent.name for agent in self.agents]

    def __bool__(self) -> bool:
        return bool(self.agents)

    def __len__(self) -> int:
        return len(self.agents)


def _read(path: Path) -> str:
    try:
        return read_text(path)
    except (ToolError, OSError):
        return ""


def _load(directory: Path, source: str) -> tuple[Subagent | None, str]:
    """One subagent directory, or the reason it is not one."""
    text = _read(directory / FILE)
    if not text:
        return None, f"{source}: could not be read"

    meta, _ = parse_frontmatter(text)
    # The path is the name, the way it is everywhere else under agent/.
    name = directory.name.strip().lower()
    description = " ".join(meta.get("description", "").split())

    if not name or len(name) > MAX_NAME or any(c.isspace() for c in name):
        return None, f"{source}: '{name}' is not a usable subagent name"
    if not description:
        # Without one the model has nothing to choose on, and a subagent it
        # cannot choose is a subagent that never runs.
        return None, f"{source}: no 'description' in its frontmatter"
    if len(description) > MAX_DESCRIPTION:
        description = description[: MAX_DESCRIPTION - 1].rstrip() + "…"

    instructions = _read(directory / INSTRUCTIONS).strip()
    if len(instructions) > MAX_INSTRUCTIONS:
        instructions = instructions[:MAX_INSTRUCTIONS].rstrip() + "\n... [truncated]"

    return Subagent(
        name=name,
        description=description,
        path=directory,
        source=source,
        instructions=instructions,
        # Its own, not the workspace's: a reviewer that quietly inherited every
        # skill in the project would be the parent agent with a different name.
        skills=scan([directory / SKILLS], directory),
    ), ""


def discover(root, include_global: bool = True) -> Found:
    """Every subagent that applies to ``root``, least specific root first."""
    root = Path(root)
    agents: dict[str, Subagent] = {}
    problems: list[str] = []

    for directory in layout.surface(root, include_global=include_global).slot("subagents"):
        for child in sorted(directory.iterdir()):
            if not (child / FILE).is_file():
                continue
            agent, problem = _load(child, layout.label(child, root))
            if agent is None:
                problems.append(problem)
            else:
                # Later root wins, as everywhere else: a workspace reviewer is
                # the one meant over a personal one of the same name.
                agents[agent.name] = agent

    return Found(agents=sorted(agents.values(), key=lambda a: a.name), problems=problems)


def roster(found: Found) -> str:
    """The subagents as the ``task`` tool describes them to the model."""
    return "\n".join(f"- {agent.name}: {agent.description}" for agent in found.agents)


def summarize(found: Found) -> str:
    """One line for the startup banner, naming what loaded and what did not."""
    parts = []
    if found.agents:
        parts.append("subagents: " + ", ".join(found.names()))
    if found.problems:
        parts.append(f"{len(found.problems)} subagent(s) skipped: " + "; ".join(found.problems))
    return "\n".join(parts)
