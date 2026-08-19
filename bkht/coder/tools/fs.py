"""Filesystem tools: read_file, list_files, write_file, edit_file.

Mutating tools are added by :func:`register_write_tools` so the read-only
build steps and ``--plan`` mode can use a registry that simply does not
contain them.
"""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError, ToolResult, Workspace, truncate

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".DS_Store",
}

MAX_READ_BYTES = 2_000_000
DEFAULT_READ_LIMIT = 400


def is_ignored(path: Path, root: Path) -> bool:
    """Whether ``path`` sits under a directory we never show the model."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in IGNORED_DIRS for part in parts)


def read_text(path: Path) -> str:
    """Read a file as text, refusing binaries rather than corrupting context."""
    if not path.exists():
        raise ToolError(f"file not found: {path.name}")
    if path.is_dir():
        raise ToolError(f"'{path.name}' is a directory, not a file")
    if path.stat().st_size > MAX_READ_BYTES:
        raise ToolError(f"file is too large to read ({path.stat().st_size} bytes)")

    data = path.read_bytes()
    if b"\x00" in data:
        raise ToolError(f"'{path.name}' looks like a binary file")
    return data.decode("utf-8", errors="replace")


def register_read_tools(registry, workspace: Workspace):
    """Add the non-mutating filesystem tools to ``registry``."""

    def read_file(path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> ToolResult:
        target = workspace.resolve(path)
        text = read_text(target)
        lines = text.splitlines()

        if offset < 1:
            raise ToolError("offset is 1-based; the first line is 1")
        if limit < 1:
            raise ToolError("limit must be at least 1")

        window = lines[offset - 1 : offset - 1 + limit]
        if not window:
            if not lines:
                return ToolResult.success(f"{workspace.relative(target)} is empty")
            raise ToolError(
                f"offset {offset} is past the end of the file ({len(lines)} lines)"
            )

        # Line numbers are shown so findings and edits can cite file:line.
        width = len(str(offset + len(window) - 1))
        body = "\n".join(
            f"{offset + i:>{width}}\t{line}" for i, line in enumerate(window)
        )

        shown_to = offset + len(window) - 1
        if shown_to < len(lines):
            body += f"\n[showing lines {offset}-{shown_to} of {len(lines)}]"
        return ToolResult.success(truncate(body))

    registry.add(
        Tool(
            name="read_file",
            description=(
                "Read a text file from the workspace. Returns numbered lines. "
                "Use offset and limit to page through a long file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "offset": {"type": "integer", "description": "First line to read, 1-based."},
                    "limit": {"type": "integer", "description": "How many lines to read."},
                },
                "required": ["path"],
            },
            run=read_file,
        )
    )

    def list_files(path: str = ".", depth: int = 2) -> ToolResult:
        root = workspace.resolve(path)
        if not root.exists():
            raise ToolError(f"path not found: {path}")
        if not root.is_dir():
            raise ToolError(f"'{path}' is a file, not a directory")
        if depth < 1:
            raise ToolError("depth must be at least 1")

        entries = list(walk(root, workspace.root, depth))
        if not entries:
            return ToolResult.success(f"{workspace.relative(root)}/ is empty")

        listing = "\n".join(entries)
        return ToolResult.success(truncate(f"{workspace.relative(root)}/\n{listing}"))

    registry.add(
        Tool(
            name="list_files",
            description=(
                "List files and directories under a path, up to a given depth. "
                "Ignores .git, node_modules, virtualenvs and build output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list. Defaults to the workspace root."},
                    "depth": {"type": "integer", "description": "How many levels deep to descend. Defaults to 2."},
                },
                "required": [],
            },
            run=list_files,
        )
    )

    return registry


def walk(directory: Path, root: Path, depth: int, prefix: str = "") -> list[str]:
    """Indented listing of ``directory``, ``depth`` levels deep."""
    if depth < 1:
        return []

    try:
        children = sorted(
            directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
    except PermissionError:
        return []

    lines: list[str] = []
    for child in children:
        if child.name in IGNORED_DIRS or child.name.startswith(".") and child.is_dir():
            continue
        if child.is_dir():
            lines.append(f"{prefix}  {child.name}/")
            lines.extend(walk(child, root, depth - 1, prefix + "  "))
        else:
            lines.append(f"{prefix}  {child.name}")
    return lines
