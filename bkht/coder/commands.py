"""Slash commands the user writes as files.

A prompt that gets retyped every day is a prompt that belongs in a file. This
is the smallest mechanism that does that: a Markdown file whose body becomes
the task, named by the file.

    ~/.bkht-coder/commands/review-tests.md   ->  /review-tests

It is deliberately not a scripting surface. The body is prose sent to the
model, `$ARGUMENTS` is the one substitution, and nothing here can run anything
-- a slash command that could execute would be a permission gate with a
back door.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .session import STATE_DIR
from .skills import parse_frontmatter
from .tools.base import ToolError
from .tools.fs import read_text

GLOBAL_ROOT = STATE_DIR / "commands"
WORKSPACE_ROOT = Path(".bkht-coder") / "commands"

PLACEHOLDER = "$ARGUMENTS"


@dataclass(frozen=True)
class UserCommand:
    """One command file: what to call it, and what it says."""

    name: str
    body: str
    description: str
    source: str

    def expand(self, argument: str) -> str:
        """The task text for this invocation.

        Substituted where the file asks for it, appended where it does not --
        so a file written without a placeholder still takes arguments rather
        than silently discarding them.
        """
        argument = argument.strip()
        if PLACEHOLDER in self.body:
            return self.body.replace(PLACEHOLDER, argument)
        return f"{self.body}\n\n{argument}".rstrip() if argument else self.body


def _read(path: Path, source: str) -> UserCommand | None:
    """One file as a command, or None if there is nothing usable in it."""
    try:
        text = read_text(path)
    except (ToolError, OSError):
        return None

    # Frontmatter is optional and only its description is used, so a command
    # file borrowed from another agent works unchanged.
    meta, body = parse_frontmatter(text)
    body = body.strip()
    if not body:
        return None

    description = " ".join(meta.get("description", "").split())
    if not description:
        description = body.splitlines()[0][:80]

    return UserCommand(name=path.stem.lower(), body=body, description=description, source=source)


def _label(path: Path, root: Path) -> str:
    for base, prefix in ((root, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def discover(root, include_global: bool = True) -> dict[str, UserCommand]:
    """Every command file that applies to ``root``, keyed by name.

    Global first, so a workspace can shadow a personal command with its own.
    """
    root = Path(root)
    directories = []
    if include_global:
        directories.append(GLOBAL_ROOT)
    directories.append(root / WORKSPACE_ROOT)

    found: dict[str, UserCommand] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            command = _read(path, _label(path, root))
            if command is not None:
                found[command.name] = command
    return found


def summarize(commands: dict[str, UserCommand]) -> str:
    """The section `/help` grows when a workspace has command files."""
    if not commands:
        return ""
    lines = ["", "Your commands"]
    lines += [
        f"  /{command.name:<18}{command.description}"
        for command in sorted(commands.values(), key=lambda c: c.name)
    ]
    return "\n".join(lines)
