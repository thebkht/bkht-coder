"""The ``agent/`` directory: one tree, one rule for what a file means.

Everything a user writes for this agent used to land somewhere different by a
rule of its own -- instructions at the workspace root, skills in three scanned
directories, slash commands in a fourth, hooks inside a JSON file. Each was
defensible on its own and the set of them was six paragraphs of README.

This is the other shape, borrowed from eve: **the slot a file lands in decides
how it loads**, and its name comes from its path. ``agent/skills/releasing/``
is the skill ``releasing`` because of where it is, not because something
declared it.

    agent/
    ├── agent.json          the marker; `{}` is enough
    ├── instructions.md     or instructions/*.md, composed in order
    ├── skills/
    ├── commands/
    ├── hooks/<event>/
    ├── subagents/<id>/
    └── tools/

Two roots, least specific first: ``~/.bkht-coder/agent/`` applies everywhere,
and ``<workspace>/agent/`` applies to one project and wins where the two
collide. Both layer *over* the older paths rather than replacing them --
``AGENTS.md``, ``.bkht-coder/skills`` and the rest keep working exactly as they
did, and a workspace that has never heard of this file is unaffected by it.

The marker is why the workspace root needs a file in it and the global one does
not. ``agent/`` is precisely the directory an eve project uses for its own
agent: without a marker, starting coder inside one would adopt that project's
system prompt as ours and import its Python into our tool registry. So an
``agent/`` we were not given is passed over in silence -- it belongs to
somebody else -- and ``agent.json`` is how a user says this one is ours. The
global root lives inside our own state directory, where that question cannot
arise.

Problems are carried rather than raised, the way :mod:`~bkht.coder.skills`
carries them. A malformed surface must cost the user their surface and a
printed sentence, never the session that would let them fix it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .session import STATE_DIR

#: The directory itself, at the workspace root and inside the state directory.
DIRECTORY = "agent"
#: What makes a workspace ``agent/`` ours. Its contents are not read yet; that
#: it parses as an object is the whole of the contract, which leaves room for
#: it to carry per-workspace settings later without breaking a file written
#: today.
MARKER = "agent.json"

GLOBAL_ROOT = STATE_DIR / DIRECTORY

#: Which of the two roots a slot directory came from. Callers layering their
#: own older paths under this one need to know: a personal ``agent/`` belongs
#: beside the other personal files, and a workspace one belongs after
#: everything, where the most specific source goes.
GLOBAL = "global"
WORKSPACE = "workspace"

#: Every slot, in the order `coder info` prints them: what the agent is told,
#: then what it can be asked for, then what runs on its own.
SLOTS = ("instructions", "skills", "commands", "hooks", "subagents", "tools")

#: Instructions are the one slot that may be a single file instead of a
#: directory, because one file is what almost every project wants.
INSTRUCTIONS = "instructions"


@dataclass(frozen=True)
class Root:
    """One adopted ``agent/`` directory, and which of the two it is."""

    path: Path
    scope: str

    def slot(self, name: str) -> Path | None:
        directory = self.path / name
        return directory if directory.is_dir() else None


@dataclass(frozen=True)
class Surface:
    """The ``agent/`` roots that apply here, least specific first."""

    roots: list[Root] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.roots)

    def slot(self, name: str, scope: str | None = None) -> list[Path]:
        """Every existing directory for one slot, least specific first.

        ``scope`` narrows it to one of the two roots, which is what a caller
        with older paths of its own needs: it already layers a personal
        directory under a workspace one, and the two halves of this surface
        belong on either side of that, not bolted onto one end.
        """
        return [
            directory
            for root in self.roots
            if (scope is None or root.scope == scope) and (directory := root.slot(name))
        ]


def _adopt(root: Path) -> tuple[bool, str]:
    """Whether a workspace ``agent/`` is ours, and why not when it is not.

    Absent or unmarked is not a problem: those are the common case and the eve
    case respectively, and neither is anything the user asked us about. A
    marker that is present and broken *is* a problem -- that file exists
    because somebody meant this to work.
    """
    marker = root / MARKER
    if not marker.is_file():
        return False, ""
    try:
        parsed = json.loads(marker.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        return False, f"{DIRECTORY}/{MARKER}: could not be read ({exc})"
    if not isinstance(parsed, dict):
        return False, f"{DIRECTORY}/{MARKER}: must be a JSON object, such as {{}}"
    return True, ""


def surface(root, include_global: bool = True) -> Surface:
    """The agent surface for a workspace at ``root``."""
    roots: list[Path] = []
    problems: list[str] = []

    if include_global and GLOBAL_ROOT.is_dir():
        roots.append(Root(GLOBAL_ROOT, GLOBAL))

    workspace = Path(root) / DIRECTORY
    if workspace.is_dir():
        adopted, problem = _adopt(workspace)
        if problem:
            problems.append(problem)
        if adopted:
            roots.append(Root(workspace, WORKSPACE))

    return Surface(roots=roots, problems=problems)


def label(path: Path, root) -> str:
    """How a path is named in a listing: short, but traceable to a file.

    The same two bases every other discovery in this package uses, so a path
    printed by `coder info` is a path the user can pass straight to an editor.
    """
    for base, prefix in ((Path(root), ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def entries(directory: Path, name: str) -> list[Path]:
    """What one slot directory holds, in the shape that slot is written in.

    Only ``hooks`` nests, and only one level: the event is the directory and
    the commands are the files inside it. Everything else is a flat listing of
    whatever is there -- a file, a directory, or something that will turn out
    to be neither, which is the slot's own discovery to report rather than
    this listing's to hide.
    """
    if not directory.is_dir():
        return []
    if name == "hooks":
        return sorted(
            child
            for event in sorted(directory.iterdir()) if event.is_dir()
            for child in event.iterdir()
        )
    return sorted(directory.iterdir())


def inventory(found: Surface, root) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    """``(root, [(slot, entries)])`` for every adopted root, for printing.

    Filesystem only. What a slot's own discovery makes of these files -- which
    skill was skipped for want of a description, which hook is missing its
    execute bit -- is that module's answer to give, and `coder info` asks each
    of them in turn.
    """
    listing = []
    for base in (root.path for root in found.roots):
        slots = []
        for name in SLOTS:
            if name == INSTRUCTIONS:
                paths = instructions(base)
                if not paths:
                    continue
            elif not (base / name).is_dir():
                # A slot nobody wrote. Distinct from an empty one, which is
                # listed: a directory the user made and has not filled is part
                # of what they meant, and a listing that hides it reads as
                # though the directory were spelled wrong.
                continue
            else:
                paths = entries(base / name, name)
            slots.append((name, [label(path, base) for path in paths]))
        listing.append((label(base, root), slots))
    return listing


def instructions(base: Path) -> list[Path]:
    """The instruction files under one root, in the order they compose.

    A flat ``instructions.md`` or a directory of them, never both: a project
    that has grown the directory has said where its instructions live, and
    reading a leftover file beside it would be reading a draft.
    """
    directory = base / INSTRUCTIONS
    if directory.is_dir():
        return sorted(path for path in directory.glob("*.md") if path.is_file())
    flat = base / f"{INSTRUCTIONS}.md"
    return [flat] if flat.is_file() else []


def render(found: Surface, root) -> str:
    """The surface as `coder info` and `/agent` print it."""
    lines = []
    if not found:
        lines.append(f"No {DIRECTORY}/ surface found.")
        lines.append(
            f"  A workspace one is {DIRECTORY}/ with a {MARKER} in it; "
            f"a personal one is {label(GLOBAL_ROOT, root)}."
        )
    for base, slots in inventory(found, root):
        lines.append(base)
        if not slots:
            lines.append("  (no slots)")
        for name, found_entries in slots:
            lines.append(f"  {name}")
            lines.extend(f"    {entry}" for entry in found_entries)

    if found.problems:
        lines.append("")
        lines.extend(f"  problem: {problem}" for problem in found.problems)
    return "\n".join(lines)
