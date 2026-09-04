"""Tool protocol, argument validation, and result type.

Every tool returns a :class:`ToolResult`; errors become text the model can act
on rather than exceptions raised through the loop. A small model recovers from
a clear error message surprisingly often, and never from a traceback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MAX_OUTPUT_LINES = 400
# The ceiling, not the working limit. 30,000 characters is roughly 7,500 tokens,
# which is 92% of the 8,192-token window this runs in by default -- a cap larger
# than the thing it exists to protect. The working limit is a share of the real
# window, resolved once at startup by :func:`set_output_budget`.
MAX_OUTPUT_CHARS = 30_000
# A quarter of the window. One tool result is allowed to be expensive, because
# reading a file is the point; it is not allowed to be the whole conversation.
OUTPUT_SHARE = 0.25
CHARS_PER_TOKEN = 4
# Below this a result is too clipped to answer anything, and a tighter cap would
# cost more turns in re-reads than it saves in tokens.
MIN_OUTPUT_CHARS = 2_000

_output_chars = MAX_OUTPUT_CHARS


def output_budget(num_ctx: int) -> int:
    """How many characters one tool result may contribute, given ``num_ctx``."""
    if not num_ctx:
        return MAX_OUTPUT_CHARS
    share = int(num_ctx * OUTPUT_SHARE) * CHARS_PER_TOKEN
    return max(MIN_OUTPUT_CHARS, min(MAX_OUTPUT_CHARS, share))


def set_output_budget(num_ctx: int) -> int:
    """Size the tool-output cap to the context window, once, at startup.

    Module state rather than a threaded parameter: ``truncate`` is called from
    six tools in five modules, the value is a process-wide constant derived from
    another process-wide constant, and threading it would put a context-window
    argument into the signature of every tool that prints anything.
    """
    global _output_chars
    _output_chars = output_budget(num_ctx)
    return _output_chars


def output_chars() -> int:
    """The cap currently in force. ``set_output_budget(0)`` restores the default."""
    return _output_chars


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

    missing = sorted(key for key in required if key not in arguments)
    unknown = sorted(key for key in arguments if key not in properties)

    # Reported together: a model that writes `filename` for `path` produces both
    # at once, and naming only one of them makes the correction a guess.
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing required argument(s) {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown argument(s) {', '.join(unknown)}")
        raise ToolError(
            f"{tool.name}: {'; '.join(problems)}. "
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


def truncate(
    text: str, max_lines: int = MAX_OUTPUT_LINES, max_chars: int | None = None
) -> str:
    """Bound tool output so one ``cat`` of a huge file cannot blow the context.

    Truncation is always announced -- a silently shortened file reads to the
    model as the whole file, and it will draw conclusions from the missing part.
    """
    limit = max_chars if max_chars is not None else _output_chars
    if len(text) > limit:
        kept = text[:limit]
        dropped = len(text) - len(kept)
        text = f"{kept}\n[truncated {dropped} characters]"

    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    dropped = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"[truncated {dropped} lines]"])


def contain(root: Path, candidate: Path) -> Path | None:
    """``candidate`` resolved, or None if it lands outside ``root``.

    The one place a sandbox boundary is decided, so every directory the agent
    is confined to -- the workspace, a skill's own folder -- is confined the
    same way. ``realpath`` follows symlinks, so a link pointing out is caught
    rather than followed.
    """
    root = Path(os.path.realpath(root))
    resolved = Path(os.path.realpath(candidate))
    if resolved != root and root not in resolved.parents:
        return None
    return resolved


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

        resolved = contain(self.root, candidate)
        if resolved is None:
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
class Reads:
    """Which files this session has read, and how they looked at the time.

    `edit_file` is the only thing that asks. An exact-string match is not on
    its own evidence that the model read anything: a string it remembered from
    an earlier session, or reconstructed from the file's name, either matches
    or it does not. When it does not, the turn spends three iterations finding
    that out -- edit, `old_string was not found`, read, edit. When it does, the
    edit lands somewhere nobody looked.

    The mtime is kept for the second half of the same problem. A file the agent
    read ten minutes ago and a human has saved since is a file whose remembered
    text is no longer what surrounds the edit, and an exact match against the
    part that did not move is exactly how that goes unnoticed.

    A path that cannot be stat'd is recorded as read anyway, with no mtime.
    Losing the ability to edit a file is a worse failure than not noticing it
    moved, and the read genuinely happened.
    """

    seen: dict[Path, float | None] = field(default_factory=dict)

    def note(self, path: Path) -> None:
        """Record ``path`` as read, as it is right now."""
        path = Path(path)
        try:
            self.seen[path] = path.stat().st_mtime
        except OSError:
            self.seen[path] = None

    def complaint(self, path: Path, shown: str) -> str:
        """What is wrong with editing ``path`` from memory, or "" if nothing is.

        ``shown`` is how the path is named to the model, so the sentence it
        gets back names the file the way it asked for it.
        """
        path = Path(path)
        # A file that is not there has a better error than this one waiting for
        # it two lines further on. "You have not read it" is true and useless.
        if not path.exists():
            return ""
        if path not in self.seen:
            return (
                f"you have not read {shown} in this session. `old_string` has to "
                "match the file exactly, and the file is the only place that "
                "text is -- read it with `read_file` first, then copy the lines "
                "you mean out of what comes back."
            )

        remembered = self.seen[path]
        if remembered is None:
            return ""
        try:
            current = path.stat().st_mtime
        except OSError:
            return ""
        if current == remembered:
            return ""
        return (
            f"{shown} has changed since you read it, so what you remember of it "
            "is not what is there now. Read it again before editing, or your "
            "edit lands in a file you have not seen."
        )


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
