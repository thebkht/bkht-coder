"""Permission policy for mutating tools.

This is the safety net that makes it acceptable to let a weak model drive. The
model proposes; the policy decides. A denial is returned as data rather than
raised, so the loop feeds it back and the model can try something else.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ASK = "ask"
AUTO = "auto"
PLAN = "plan"
MODES = (ASK, AUTO, PLAN)

MAX_PREVIEW_LINES = 40


@dataclass
class Decision:
    """Whether a call may proceed, and why not if it may not."""

    allowed: bool
    reason: str = ""


def preview(tool, arguments: dict, workspace) -> str:
    """What the user is being asked to approve.

    Showing the actual diff rather than the tool name is the point: approving
    'edit_file' blind is not meaningfully different from running in --auto.
    """
    if tool.name == "bash":
        return f"$ {arguments.get('command', '')}"

    path = arguments.get("path")
    if path is None:
        return ""

    try:
        target = workspace.resolve(path)
    except Exception:
        return str(path)

    before = ""
    if target.exists() and target.is_file():
        try:
            before = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            before = ""

    if tool.name == "write_file":
        after = arguments.get("content", "")
    elif tool.name == "edit_file":
        old, new = arguments.get("old_string", ""), arguments.get("new_string", "")
        if arguments.get("replace_all"):
            after = before.replace(old, new)
        else:
            after = before.replace(old, new, 1)
    else:
        return workspace.relative(target)

    label = workspace.relative(target)
    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=label,
            tofile=label,
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return f"{label} (no change)"
    if len(diff) > MAX_PREVIEW_LINES:
        dropped = len(diff) - MAX_PREVIEW_LINES
        diff = diff[:MAX_PREVIEW_LINES] + [f"[{dropped} more diff lines]"]
    return "\n".join(diff)


def ask_terminal(question: str, body: str) -> str:
    """Default prompt: show the change, then ask. Returns y / n / a."""
    if body:
        print(body)
    try:
        return input(f"{question} [y/N/a=always] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "n"


@dataclass
class Permissions:
    """Gate on mutating tool calls.

    ``ask``  -- prompt before each mutating call, showing the diff or command.
    ``auto`` -- allow everything, for scripted runs.
    ``plan`` -- deny every mutation, for read-only exploration.
    """

    mode: str = ASK
    workspace: object = None
    prompt: Callable[[str, str], str] = ask_terminal
    always_allow: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown permission mode {self.mode!r}; expected one of {', '.join(MODES)}")

    def check(self, tool, arguments: dict) -> Decision:
        """Decide whether ``tool`` may run with ``arguments``."""
        if not tool.mutating:
            return Decision(True)

        if self.mode == AUTO:
            return Decision(True)

        if self.mode == PLAN:
            return Decision(
                False,
                f"{tool.name} would change the workspace, and this session is in "
                "plan mode. Describe the change instead of making it.",
            )

        if tool.name in self.always_allow:
            return Decision(True)

        body = preview(tool, arguments, self.workspace) if self.workspace else ""
        answer = self.prompt(f"Allow {tool.name}?", body)

        if answer in ("a", "always"):
            self.always_allow.add(tool.name)
            return Decision(True)
        if answer in ("y", "yes"):
            return Decision(True)
        return Decision(
            False,
            f"the user declined this {tool.name} call. Do not retry it; either "
            "propose a different approach or explain what you would have done.",
        )
