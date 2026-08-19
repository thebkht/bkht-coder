"""System and task prompts.

Because tool calls travel in message *content* rather than the ``tool_calls``
field, the model is responsible for formatting them. The system prompt
therefore states the emission format explicitly and shows an example -- this is
not decoration, it is the wire protocol.
"""

from __future__ import annotations

import json

TOOL_PROTOCOL = """\
# Calling a tool

To use a tool, reply with ONLY a JSON object, on its own, in this exact shape:

{"name": "<tool name>", "arguments": {<arguments>}}

For example, to read a file:

{"name": "read_file", "arguments": {"path": "src/main.py"}}

Rules:
- Emit the JSON object and nothing else. No explanation before or after it.
- One tool call per reply. Wait for the result before calling the next tool.
- Use exactly the argument names listed for that tool.
- The result comes back as a `tool` message. Read it before deciding what to do.

When you have finished the task, reply with a normal answer in plain prose and
no JSON. That is how you signal you are done."""


def describe_tools(tools) -> str:
    """The tool list as it appears in the system prompt.

    Arguments are spelled out per tool rather than dumped as raw JSON Schema:
    a small model follows a short readable list far more reliably.
    """
    blocks = []
    for tool in tools:
        properties = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))

        args = []
        for name, spec in properties.items():
            marker = "" if name in required else " (optional)"
            description = spec.get("description", "")
            args.append(f"    - {name}: {spec.get('type', 'any')}{marker} — {description}")

        block = f"## {tool.name}\n{tool.description}\nArguments:"
        block += "\n" + ("\n".join(args) if args else "    (none)")
        if tool.mutating:
            block += "\n  This tool changes files and needs permission."
        blocks.append(block)

    return "\n\n".join(blocks)


SYSTEM = """\
You are `coder`, a coding agent working in a single directory on the user's machine.

You answer questions about the code and make changes to it, by calling tools.
Work from what you actually read: look at the real files before making a claim
about them. Never invent a file path, a function name, or a line of code.

Keep going until the task is done. Prefer several small, verified steps over one
large guess. When you are unsure which file matters, use `glob` and `grep` to
find out rather than assuming.

# Workspace

Root: {root}
All paths are relative to that root. You cannot read or write outside it.

{tree}

# Tools

{tools}

{protocol}"""


def system_prompt(registry, root: str, tree: str = "") -> str:
    """Assemble the system prompt for a session."""
    tree_block = f"Files:\n{tree}" if tree else ""
    return SYSTEM.format(
        root=root,
        tree=tree_block,
        tools=describe_tools(registry),
        protocol=TOOL_PROTOCOL,
    )


def malformed_call(error: str, tool_names: list[str]) -> str:
    """The corrective message sent back after a schema violation.

    It restates the protocol, because the most common failure on a small model
    is drifting away from the format rather than misunderstanding the task.
    """
    return (
        f"Your tool call was not valid: {error}\n\n"
        f"Available tools: {', '.join(tool_names)}.\n"
        "Reply with a single JSON object and nothing else, in the form\n"
        '{"name": "<tool name>", "arguments": {<arguments>}}\n'
        "Correct the call and try again, or answer in plain prose if you are done."
    )


def no_call_and_no_answer(tool_names: list[str]) -> str:
    """Sent when a reply contains neither a usable call nor a plausible answer."""
    return (
        "That reply contained neither a tool call nor an answer.\n"
        f"Available tools: {', '.join(tool_names)}.\n"
        "Either call a tool with a single JSON object in the form\n"
        '{"name": "<tool name>", "arguments": {<arguments>}}\n'
        "or give your final answer in plain prose."
    )


def tool_schema_hint(tool) -> str:
    """A single tool's schema, for when the model needs the exact shape."""
    return json.dumps(tool.parameters, indent=2)
