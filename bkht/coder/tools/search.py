"""Search tools: grep and glob.

Implemented in pure Python rather than shelling out to ripgrep, so the agent
has no dependency on what happens to be installed on the machine.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .base import Tool, ToolError, ToolResult, Workspace, truncate
from .fs import is_ignored, read_text

MAX_MATCHES = 200
MAX_SCANNED_BYTES = 1_000_000


def iter_files(root: Path, workspace_root: Path, pattern: str | None = None):
    """Every non-ignored file under ``root``, optionally filtered by a glob."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or is_ignored(path, workspace_root):
            continue
        if pattern and not _matches(path, workspace_root, pattern):
            continue
        yield path


def _matches(path: Path, root: Path, pattern: str) -> bool:
    """Whether ``path`` matches ``pattern``, tolerating the common spellings.

    A small model writes ``*.py`` as often as ``**/*.py`` when it means "any
    Python file anywhere", so both are accepted.
    """
    relative = str(path.relative_to(root)) if root in path.parents else path.name
    return (
        fnmatch.fnmatch(relative, pattern)
        or fnmatch.fnmatch(path.name, pattern)
        or fnmatch.fnmatch(relative, f"**/{pattern}")
    )


def register_search_tools(registry, workspace: Workspace):
    """Add grep and glob to ``registry``."""

    def grep(pattern: str, path: str = ".", glob: str | None = None) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression {pattern!r}: {exc}") from None

        root = workspace.resolve(path)
        if not root.exists():
            raise ToolError(f"path not found: {path}")

        targets = [root] if root.is_file() else iter_files(root, workspace.root, glob)

        matches: list[str] = []
        for file in targets:
            if file.stat().st_size > MAX_SCANNED_BYTES:
                continue
            try:
                text = read_text(file)
            except ToolError:
                continue  # binary or unreadable; not an error for a search

            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        f"{workspace.relative(file)}:{number}: {line.strip()[:200]}"
                    )
                    if len(matches) >= MAX_MATCHES:
                        matches.append(f"[stopped after {MAX_MATCHES} matches]")
                        return ToolResult.success("\n".join(matches))

        if not matches:
            scope = f" in {path}" if path != "." else ""
            return ToolResult.success(f"No matches for {pattern!r}{scope}.")
        return ToolResult.success(truncate("\n".join(matches)))

    registry.add(
        Tool(
            name="grep",
            description=(
                "Search file contents with a regular expression. Returns "
                "file:line: matched-text. Use glob to limit which files are scanned."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "File or directory to search. Defaults to the workspace root."},
                    "glob": {"type": "string", "description": "Only search files matching this glob, e.g. *.py"},
                },
                "required": ["pattern"],
            },
            run=grep,
        )
    )

    def glob_tool(pattern: str) -> ToolResult:
        found = [
            workspace.relative(path)
            for path in iter_files(workspace.root, workspace.root, pattern)
        ]
        if not found:
            return ToolResult.success(f"No files match {pattern!r}.")
        return ToolResult.success(truncate("\n".join(found)))

    registry.add(
        Tool(
            name="glob",
            description=(
                "Find files by name pattern, e.g. *.py or src/**/*.ts. "
                "Returns paths relative to the workspace root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match against file paths."},
                },
                "required": ["pattern"],
            },
            run=glob_tool,
        )
    )

    return registry
