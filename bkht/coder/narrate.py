"""Saying what a tool call is for, in words.

A transcript of `edit_file(path=bkht/coder/cli.py, old_string=..., new_string=...)`
is a log of function calls; what a reader actually wants to know is that the
agent is editing cli.py. The call itself is printed once it has finished, with
the mark that says whether it worked; this sentence is what the status line
says while it is still running, and for a long call it is the only thing
saying what the agent is doing at all.

Deliberately mechanical: this describes the call, it does not ask the model to
narrate itself. A second opinion about its own intentions would cost a round
trip and could differ from what it then did.
"""

from __future__ import annotations

from .parsing import ToolCall

FALLBACK = "Working"


def _short(value, limit: int = 60) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def intent(call: ToolCall) -> str:
    """One line saying what this call is about to do."""
    arguments = call.arguments or {}
    path = arguments.get("path")

    if call.name == "read_file":
        return f"Reading {_short(path)}" if path else "Reading a file"
    if call.name == "write_file":
        return f"Writing {_short(path)}" if path else "Writing a file"
    if call.name == "edit_file":
        return f"Editing {_short(path)}" if path else "Editing a file"
    if call.name == "list_files":
        return f"Listing {_short(path)}" if path else "Listing the workspace"
    if call.name == "grep":
        pattern = arguments.get("pattern")
        return f"Searching for {_short(pattern)}" if pattern else "Searching the workspace"
    if call.name == "glob":
        pattern = arguments.get("pattern")
        return f"Looking for files matching {_short(pattern)}" if pattern else "Looking for files"
    if call.name == "codebase_search":
        terms = arguments.get("terms")
        return f"Looking for {_short(terms)}" if terms else "Searching the workspace"
    if call.name == "bash":
        command = arguments.get("command")
        return f"Running {_short(command)}" if command else "Running a command"

    # An unknown tool -- a future one, or one a model invented -- still gets a
    # sentence rather than nothing.
    return f"Calling {call.name}" if call.name else FALLBACK
