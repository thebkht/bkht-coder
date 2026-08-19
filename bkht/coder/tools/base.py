"""Tool protocol, argument validation, and result type.

Every tool returns a :class:`ToolResult`; errors become text the model can act
on rather than exceptions raised through the loop. A small model recovers from
a clear error message surprisingly often, and never from a traceback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

MAX_OUTPUT_LINES = 400
MAX_OUTPUT_CHARS = 30_000


class ToolError(Exception):
    """A tool failed in a way the model should be told about and can retry."""


@dataclass
class ToolResult:
    """The outcome of one tool call."""

    ok: bool
    content: str = ""
    error: str = ""

    @classmethod
    def success(cls, content: str) -> "ToolResult":
        return cls(ok=True, content=content)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(ok=False, error=error)

    def as_message(self) -> str:
        """What gets fed back to the model as the tool's output."""
        return self.content if self.ok else f"ERROR: {self.error}"


@dataclass
class Tool:
    """A callable the model may invoke, plus the schema it is validated against.

    ``parameters`` is a JSON Schema object. Validation is done here rather than
    inside each tool so that every schema violation produces the same shape of
    corrective message.
    """

    name: str
    description: str
    parameters: dict
    run: Callable[..., ToolResult]
    mutating: bool = False

    def declaration(self) -> dict:
        """The tool as declared to the provider and in the system prompt."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def validate_arguments(tool: Tool, arguments: dict) -> dict:
    """Check ``arguments`` against ``tool``'s schema, returning coerced values.

    Raises :class:`ToolError` with a message naming the specific violation, so
    the model can correct itself on the retry rather than guessing.
    """
    if not isinstance(arguments, dict):
        raise ToolError(
            f"{tool.name}: arguments must be a JSON object, got {type(arguments).__name__}"
        )

    schema = tool.parameters
    properties: dict = schema.get("properties", {})
    required: list = schema.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolError(
            f"{tool.name}: missing required argument(s) {', '.join(sorted(missing))}. "
            f"Expected: {', '.join(properties) or 'none'}"
        )

    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise ToolError(
            f"{tool.name}: unknown argument(s) {', '.join(sorted(unknown))}. "
            f"Expected: {', '.join(properties) or 'none'}"
        )

    coerced: dict[str, Any] = {}
    for key, value in arguments.items():
        expected = properties[key].get("type")
        allowed = _TYPES.get(expected)
        if allowed is None:
            coerced[key] = value
            continue

        # bool is a subclass of int, so an explicit guard is needed both ways.
        if expected in ("integer", "number") and isinstance(value, bool):
            raise ToolError(f"{tool.name}: argument '{key}' must be a {expected}")
        if expected == "boolean" and not isinstance(value, bool):
            raise ToolError(f"{tool.name}: argument '{key}' must be a boolean")

        if not isinstance(value, allowed):
            # Small models routinely stringify numbers; accept that narrowly.
            if expected in ("integer", "number") and isinstance(value, str):
                try:
                    value = int(value) if expected == "integer" else float(value)
                except ValueError:
                    raise ToolError(
                        f"{tool.name}: argument '{key}' must be a {expected}, "
                        f"got {value!r}"
                    ) from None
            else:
                raise ToolError(
                    f"{tool.name}: argument '{key}' must be a {expected}, "
                    f"got {type(value).__name__}"
                )
        coerced[key] = value

    return coerced


def truncate(text: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Bound tool output so one ``cat`` of a huge file cannot blow the context.

    Truncation is always announced -- a silently shortened file reads to the
    model as the whole file, and it will draw conclusions from the missing part.
    """
    if len(text) > MAX_OUTPUT_CHARS:
        kept = text[:MAX_OUTPUT_CHARS]
        dropped = len(text) - len(kept)
        text = f"{kept}\n[truncated {dropped} characters]"

    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    dropped = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"[truncated {dropped} lines]"])


@dataclass
class Workspace:
    """The directory the agent is confined to.

    Every path a tool touches goes through :meth:`resolve`, which rejects
    absolute paths outside the root and any ``..`` escape -- including escapes
    that only appear after symlinks are followed.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    def resolve(self, path: str) -> Path:
        """Absolute path for ``path``, or :class:`ToolError` if it escapes."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path must be a non-empty string")

        candidate = Path(os.path.expanduser(path))
        if not candidate.is_absolute():
            candidate = self.root / candidate

        # resolve() follows symlinks, so a symlink pointing outside is caught.
        resolved = Path(os.path.realpath(candidate))
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(
                f"path '{path}' is outside the workspace root {self.root}"
            )
        return resolved

    def relative(self, path: Path) -> str:
        """``path`` shown relative to the root, for display and findings."""
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return str(path)


@dataclass
class Registry:
    """Name -> Tool, kept small on purpose.

    Every extra tool measurably degrades selection accuracy on a small model,
    so this is a curated set rather than a plugin surface.
    """

    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> Tool:
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def declarations(self) -> list[dict]:
        return [self.tools[name].declaration() for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    def __iter__(self):
        return iter(self.tools[name] for name in self.names())

    def __len__(self) -> int:
        return len(self.tools)
