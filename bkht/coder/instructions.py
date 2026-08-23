"""Project instruction files: the user's standing rules for this workspace.

A coding agent that has to be told the project's conventions in every prompt is
a worse tool than one that reads them once. These files are the mechanism: drop
an ``AGENTS.md`` or ``CLAUDE.md`` in the workspace and its contents become part
of the system prompt, binding on the model for every turn.

The budget below is not tidiness. ``num_ctx`` is 16384 by default, so every
character of instruction is a character the conversation does not get. A file
that grows without bound would quietly starve the session it was meant to help.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tools.base import ToolError
from .tools.fs import read_text

# Global rules live beside the session state, in ``~/.bkht-coder``. Deliberately
# not ``~/.claude/CLAUDE.md``: that file belongs to a different tool and is not
# ours to consume.
GLOBAL_DIR = Path("~/.bkht-coder").expanduser()
GLOBAL_NAME = "AGENTS.md"

# Workspace root only. No parent walk, no per-directory nesting: both cost
# context we do not have, and give a small model more state to lose track of.
WORKSPACE_NAMES = ("AGENTS.md", "CLAUDE.md")

MAX_INSTRUCTION_CHARS = 4000
MAX_PER_SOURCE = 2000
TRUNCATION_NOTE = "... [truncated: this instruction file is too long to include in full]"


@dataclass
class Instruction:
    """One instruction file that was found, read, and fitted to the budget."""

    source: str
    text: str
    truncated: bool = False


def _label(path: Path, root: Path) -> str:
    """How a source is named in the prompt: short, but unambiguous.

    Workspace files are named relative to the root, the same way every other
    path the model sees is, so a rule can be traced back to a file the model
    can actually open.
    """
    for base, prefix in ((root, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _read(path: Path) -> str | None:
    """The file's text, or None if it is absent, empty, or unreadable.

    A malformed instruction file must not stop the agent from starting -- the
    user is then left with no way to run the tool that would fix it.
    """
    try:
        text = read_text(path)
    except (ToolError, OSError):
        return None
    return text.strip() or None


def _fit(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` to ``limit`` characters, announcing the cut if one happens."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n" + TRUNCATION_NOTE, True


def load_instructions(
    root,
    include_global: bool = True,
    budget: int = MAX_INSTRUCTION_CHARS,
    per_source: int = MAX_PER_SOURCE,
) -> list[Instruction]:
    """Every instruction file that applies to ``root``, least specific first.

    The total budget is spent from the most specific source backwards, so a
    long global file cannot crowd out the workspace's own rules -- the specific
    instruction is the one the user is more likely to mean.
    """
    root = Path(root)
    candidates: list[Path] = []
    if include_global:
        candidates.append(GLOBAL_DIR / GLOBAL_NAME)
    candidates.extend(root / name for name in WORKSPACE_NAMES)

    found = []
    for path in candidates:
        text = _read(path)
        if text is not None:
            found.append((path, *_fit(text, per_source)))

    kept: list[Instruction] = []
    remaining = budget
    for path, text, truncated in reversed(found):
        if remaining <= 0:
            break
        if len(text) > remaining:
            text, truncated = _fit(text, remaining)
            remaining = 0
        else:
            remaining -= len(text)
        kept.append(Instruction(source=_label(path, root), text=text, truncated=truncated))

    kept.reverse()
    return kept


def render(instructions: list[Instruction]) -> str:
    """The instruction block as it appears in the system prompt.

    Each block names its source so the model can attribute a rule, and so the
    user can tell which file is talking when two of them disagree.
    """
    return "\n\n".join(f"From {i.source}:\n\n{i.text}" for i in instructions)


def summarize(instructions: list[Instruction]) -> str:
    """One line naming what was loaded, for the startup banner."""
    if not instructions:
        return ""
    parts = [i.source + (" (truncated)" if i.truncated else "") for i in instructions]
    return "instructions: " + ", ".join(parts)
