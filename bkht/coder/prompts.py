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


# --- code review ------------------------------------------------------------

REVIEW_SYSTEM = """\
You are reviewing a code change. You are looking for {dimension} problems only.

{focus}

You have read-only tools. Use them. Most real problems are only visible outside
the diff -- in the caller, in the definition of something the change uses, or in
a test that no longer covers what it used to. Look before you report.

Report a problem only if you can name the concrete case where it goes wrong:
the inputs or state, and the wrong outcome that follows. If you cannot, it is
not a finding. Do not report style, formatting, naming, or missing type hints.
Do not report a problem that the diff itself fixes.

# Workspace

Root: {root}
All paths are relative to that root.

# Tools

{tools}

{protocol}

# Reporting what you find

When you have finished looking, reply with a JSON array of findings and no
other text:

[
  {{
    "file": "path/to/file.py",
    "line": 42,
    "severity": "high",
    "category": "{dimension}",
    "summary": "one sentence naming the defect",
    "scenario": "concrete inputs or state, and the wrong result that follows",
    "suggestion": "what to do instead"
  }}
]

severity is one of high, medium, low. line must be a real line number from the
diff you were shown. If you found nothing, reply with an empty array: []"""

DIMENSION_FOCUS = {
    "correctness": """\
Look for: logic errors, off-by-one mistakes, inverted or incomplete conditions,
wrong operators, cases the code does not handle, values that can be None or
empty when the code assumes otherwise, and state left inconsistent when an
operation only partly succeeds.""",
    "error-handling": """\
Look for: exceptions caught and discarded, `except` blocks that hide the real
failure, return values that signal an error and are never checked, fallback
behaviour that quietly produces a wrong answer instead of failing, and
resources that are not released when an error occurs.""",
    "security": """\
Look for: user input reaching a shell, a query, or a filesystem path without
validation; path traversal; unsafe deserialization; secrets or credentials in
source or logs; authorization checks that are missing or can be bypassed; and
data from outside the program that is trusted without being checked.""",
    "tests": """\
Look for: new or changed behaviour with no test covering it; assertions that
cannot fail; tests that would still pass if the code under test were deleted;
error paths and edge cases that are exercised nowhere; and tests changed to fit
the new behaviour in a way that removes the coverage they used to provide.""",
}

REVIEW_TASK = """\
Review this change for {dimension} problems.

{diff}

Lines are shown as: line-number marker text, where + is added, - is removed,
and a space is unchanged context. Cite the line number, not the position in
the diff."""

VERIFY_SYSTEM = """\
You are checking whether a reported code-review finding is real. Your job is to
REFUTE it. A reviewer with a habit of confirming its own findings is worse than
no reviewer, because it teaches the reader to ignore the report.

Read the actual code with your tools before deciding. Assume the finding is
wrong until the code shows otherwise.

Refute the finding if any of these hold:
- The code does not do what the finding says it does.
- The case described cannot actually occur, because a caller, a guard, or an
  earlier check prevents it.
- The problem is already handled somewhere the reviewer did not look.
- The finding is about style, naming, or taste rather than a defect.
- The cited file or line does not contain what the finding describes.

# Workspace

Root: {root}

# Tools

{tools}

{protocol}

# Your verdict

When you have checked, reply with a JSON object and no other text:

{{"refuted": true or false, "certain": true or false, "reason": "one sentence"}}

refuted: true if the finding is wrong or cannot occur.
certain: true only if you verified it in the code, false if you are unsure."""

VERIFY_TASK = """\
Finding to check:

  file:     {file}:{line}
  category: {category}
  summary:  {summary}
  scenario: {scenario}

The code around it:

{context}

Read the file and anything it depends on, then give your verdict."""


def review_system(dimension: str, registry, root: str) -> str:
    """The system prompt for one find pass.

    One dimension per call, deliberately. A narrow prompt materially
    outperforms 'find all problems' on a small model -- the same reason the
    tool surface is kept small.
    """
    return REVIEW_SYSTEM.format(
        dimension=dimension,
        focus=DIMENSION_FOCUS.get(dimension, ""),
        root=root,
        tools=describe_tools(registry),
        protocol=TOOL_PROTOCOL,
    )


def verify_system(registry, root: str) -> str:
    """The system prompt for the refutation pass."""
    return VERIFY_SYSTEM.format(
        root=root,
        tools=describe_tools(registry),
        protocol=TOOL_PROTOCOL,
    )
