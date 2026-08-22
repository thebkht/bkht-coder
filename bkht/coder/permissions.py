"""Permission policy for mutating tools.

This is the safety net that makes it acceptable to let a weak model drive. The
model proposes; the policy decides. A denial is returned as data rather than
raised, so the loop feeds it back and the model can try something else.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable

from .rules import ALLOW, DENY, Rules

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
    command = arguments.get("command", "")
    if tool.name == "background":
        # `output`, `stop` and `list` carry no command and change nothing that
        # can be previewed; the action itself is all there is to show.
        return f"$ {command} &" if command else f"background {arguments.get('action', '')}"

    if command:
        return f"$ {command}"

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
    return "\n".join(diff)


def truncate(body: str, limit: int = MAX_PREVIEW_LINES) -> str:
    """Bound a preview for display.

    Kept out of :func:`preview` because how much fits on a screen is a property
    of the screen, not of the change -- and a prompt that can expand the diff
    on a keypress needs the whole thing to expand to.
    """
    lines = body.splitlines()
    if len(lines) <= limit:
        return body
    return "\n".join(lines[:limit] + [f"[{len(lines) - limit} more diff lines]"])


def ask_terminal(question: str, body: str) -> str:
    """Default prompt: show the change, then ask. Returns y / n / a."""
    if body:
        print(truncate(body))
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
    rules: Rules | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown permission mode {self.mode!r}; expected one of {', '.join(MODES)}")
        if self.rules is None and self.workspace is not None:
            self.rules = Rules.load(str(getattr(self.workspace, "root", "")))

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

        # A decision the user already made about this exact call, in this
        # workspace. Not about this tool: approving one shell command must
        # never approve the next one.
        remembered = self.rules.decide(tool.name, arguments) if self.rules else None
        if remembered == ALLOW:
            return Decision(True)
        if remembered == DENY:
            return Decision(False, self.refusal(tool))

        body = preview(tool, arguments, self.workspace) if self.workspace else ""
        answer = self.prompt(f"Allow {tool.name}?", body)

        if answer in ("a", "always"):
            if self.rules:
                self.rules.remember(tool.name, arguments, ALLOW)
            return Decision(True)
        if answer in ("y", "yes"):
            return Decision(True)
        return Decision(False, self.refusal(tool))

    @staticmethod
    def refusal(tool) -> str:
        """What the model is told when a call is refused, however it was refused.

        A remembered denial and a fresh one read identically on purpose: the
        model has nothing useful to do with the difference, and telling it a
        rule exists invites arguing with the rule.
        """
        return (
            f"the user declined this {tool.name} call. Do not retry it; either "
            "propose a different approach or explain what you would have done."
        )
