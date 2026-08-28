"""Saying what a tool call is for, in words.

A transcript of `edit_file(path=bkht/coder/cli.py, old_string=..., new_string=...)`
is a log of function calls; what a reader actually wants to know is that the
agent is editing cli.py. The call itself is printed once it has finished, with
the mark that says whether it worked; this sentence is what the status line
says while it is still running, and for a long call it is the only thing
saying what the agent is doing at all.

What came back is said the same way, by :func:`outcome`. Until it was, the
transcript showed the call and never its result -- so a model that lost a file
to compaction and wrote out what it remembered of it produced something
indistinguishable, on screen, from a real reading of that file. A count beside
the call is what makes an invented result look invented.
"""

from __future__ import annotations

from .parsing import ToolCall

FALLBACK = "Working"

#: The result summary is a reassurance, not the output. Anything longer than
#: this is the tool's job to have truncated, not this line's job to show.
SUMMARY_LIMIT = 70


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


def _count(number: int, thing: str) -> str:
    """`3 matches`, `1 file`. English enough for the four nouns this uses."""
    if number == 1:
        return f"{number} {thing}"
    plural = f"{thing}es" if thing.endswith(("ch", "sh", "s", "x")) else f"{thing}s"
    return f"{number} {plural}"


def _lines(content: str) -> list[str]:
    return content.splitlines()


def outcome(call: ToolCall, content: str) -> str:
    """One line saying what a finished call returned.

    Empty when there is nothing worth saying: a tool whose whole output is one
    short sentence has already said it, and repeating it under itself would be
    noise where the point is signal.
    """
    lines = _lines(content)
    if not lines:
        return "nothing"

    if call.name == "grep":
        # `file:line: text`, so the file is everything before the first colon.
        files = {line.split(":", 1)[0] for line in lines if ":" in line}
        found = _count(len(lines), "match")
        return f"{found} in {_count(len(files), 'file')}" if files else found
    if call.name in ("glob", "list_files"):
        return _count(len(lines), "path")
    if call.name == "codebase_search":
        return _count(len(lines), "line")
    if call.name == "read_file":
        return _count(len(lines), "line")

    # Everything else: its own first line if that is the whole of it, and a
    # count when it is not. `edit_file` and `write_file` answer in a sentence;
    # `bash` answers in whatever the command printed.
    first = lines[0].strip()
    if len(lines) == 1:
        return first if len(first) <= SUMMARY_LIMIT else _count(len(first), "character")
    return _count(len(lines), "line")
