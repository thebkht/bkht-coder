"""The `plan` tool: write the list, tick it off.

One tool with two jobs, not two tools. The registry's own argument -- every
extra tool measurably degrades selection accuracy on a 14b -- applies to the
cure as much as the disease, and `plan(steps=...)` and `plan(done=...)` are
plainly the same verb from the model's side. Naming them separately would buy
a cleaner schema and pay for it in the one currency that matters here.

The plan lives on the session, so this module only edits it; see
:mod:`bkht.coder.plan` for why it lives there and not in the message history.
"""

from __future__ import annotations

from ..plan import MAX_STEPS
from .base import Registry, Tool, ToolError, ToolResult

DESCRIPTION = (
    "Record the steps you are going to take, and tick them off as you finish "
    "them. Write a plan before a task that needs more than one or two tool "
    "calls; your plan is shown back to you on every reply, including after "
    "earlier messages have been dropped to free room, so it is what you have "
    "left when you have lost the rest. Pass `steps` to write or rewrite the "
    "list, `done` to tick one step. Do not plan a task you can answer in a "
    "sentence."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": (
                f"The whole plan, replacing any earlier one. At most {MAX_STEPS} "
                "short lines, one per step."
            ),
            "items": {"type": "string"},
        },
        "done": {
            "type": "integer",
            "description": "Number of the step that is now finished, counting from 1.",
        },
    },
    # Neither is required on its own; that both cannot be omitted is checked in
    # `run`, because a JSON Schema `anyOf` is not something the validator here
    # reads and not something a 14b reads either.
    "required": [],
}


def register_plan_tool(registry: Registry, session) -> None:
    """Offer `plan`, writing to ``session``'s plan."""

    def run(steps=None, done=None) -> ToolResult:
        if steps is None and done is None:
            raise ToolError(
                "plan: give `steps` to write the plan, or `done` to tick a step off"
            )

        said = []
        if steps is not None:
            if not all(isinstance(line, str) for line in steps):
                raise ToolError("plan: `steps` must be a list of strings")
            # Checked against the cleaned list rather than the raw one: a model
            # that sent [""] sent a plan the tool would silently discard, and
            # the error has to name that, not congratulate it.
            if not [line for line in steps if line.strip()]:
                raise ToolError("plan: `steps` was empty; give at least one step")
            session.set_plan(steps)
            said.append(f"Plan set, {len(session.plan)} steps.")

        if done is not None:
            try:
                step = session.tick_plan(done)
            except IndexError as exc:
                # A wrong number is the ordinary mistake here, and the plan is
                # in the message below so the model can pick the right one
                # without spending another round trip finding it.
                raise ToolError(f"plan: {exc}") from None
            said.append(f"Ticked step {done}: {step.text}")

        finished, total = session.plan.progress()
        said.append(f"\n{session.plan.render()}\n\n{finished}/{total} done.")
        if session.plan.finished():
            said.append("Every step is ticked. Answer the user now, in prose.")
        return ToolResult.success(" ".join(said))

    registry.add(
        Tool(
            name="plan",
            description=DESCRIPTION,
            parameters=SCHEMA,
            run=run,
            # Nothing on disk changes, so no approval is asked for. A plan the
            # user has to approve is a plan the model stops writing.
            mutating=False,
        )
    )
