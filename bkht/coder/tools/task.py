"""The `task` tool: hand a piece of work to a second agent and keep the answer.

The README already describes the failure this is for. Reading is what fills the
window, and most reading is spent finding out where something is rather than on
the thing itself: a request to review five files costs the parent five whole
files, of which it needs perhaps thirty lines. It compacts, loses the files,
reads them again, and stops at the time cap having answered nothing.

A delegated task pays that cost somewhere else. The sub-agent gets its own
:class:`~bkht.coder.session.Session`, does the reading in a window of its own,
and hands back prose. The parent's history grows by one tool result instead of
by ten, and the search that produced it is simply gone -- which is the correct
outcome, because the parent never wanted the search.

Three bounds, all of them deliberate:

* **Read-only.** ``build_registry(read_only=True)`` -- a sub-agent cannot write
  files, run a shell, or start a job. Delegation exists to save context, and a
  nested agent making changes the user was never shown is a different feature
  with a different set of questions to answer.
* **No nesting.** The sub-agent's registry has no ``task`` tool, so a turn
  cannot fan out into a tree of agents whose cost nobody bounded.
* **Its own clock.** A share of a turn, not a turn: a delegated search that
  runs the parent's whole budget has spent the turn rather than saved it.

Esc reaches into it without anything here arranging that. ``interrupt_main``
raises ``KeyboardInterrupt`` on the thread running the loop, the sub-agent runs
on that same thread inside the tool call, and :meth:`Agent._execute` catches
``Exception`` -- so the cancellation travels up through the nested loop and out
of the parent, which is what a user pressing Esc meant.
"""

from __future__ import annotations

from pathlib import Path

from .base import Registry, Tool, ToolError, ToolResult

#: How long a delegated task may run. Short next to the parent's ten minutes:
#: this is one question, and a sub-agent still going after three of them is not
#: about to produce an answer worth the wait.
TASK_SECONDS = 180.0

#: And how many round trips. Low, because a task worth delegating is a search
#: -- find it, read it, say what it says -- and one that needs fifteen steps
#: was not a task, it was the job.
TASK_ITERATIONS = 8

DESCRIPTION = (
    "Hand one self-contained question to a second agent, which searches and "
    "reads on its own and returns a written answer. Use it when finding the "
    "answer will mean opening several files but the answer itself is short -- "
    "'which module builds the tool registry, and how', 'summarise what "
    "review/ci.py does'. The files it reads cost you nothing; only its answer "
    "comes back, so ask for everything you need in one go. It cannot change "
    "files, run commands, or delegate further. Do not use it for work you can "
    "do in a call or two yourself."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": (
                "The whole question, written for someone who cannot see this "
                "conversation. Name the files or area if you know them, and say "
                "what the answer should cover."
            ),
        }
    },
    "required": ["instruction"],
}


class Quiet:
    """Forwards a sub-agent's tool calls to the user, and swallows its prose.

    The calls are worth showing: a delegated task is the one part of a turn the
    user cannot otherwise see, and an agent that goes quiet for ninety seconds
    looks stuck. Its tokens are not -- they would stream into the middle of the
    parent's answer, which is the one place they do not belong. The parent will
    print what comes back, once, as a tool result.
    """

    def __init__(self, listener) -> None:
        self.listener = listener

    def on_token(self, text: str) -> None:
        pass

    def on_tool_call(self, call) -> None:
        self.listener.on_tool_call(call)

    def on_tool_result(self, call, result) -> None:
        self.listener.on_tool_result(call, result)

    def on_retry(self, reason: str) -> None:
        self.listener.on_retry(reason)


def register_task_tool(
    registry: Registry,
    root: Path | str,
    provider,
    skills=None,
    listener=None,
    hooks=None,
    seconds: float = TASK_SECONDS,
    iterations: int = TASK_ITERATIONS,
) -> None:
    """Offer `task`, delegating to a read-only agent under ``root``."""

    def run(instruction: str) -> ToolResult:
        if not instruction.strip():
            raise ToolError("task: `instruction` must say what to find out")

        # Imported here rather than at module scope: `agent` imports this
        # package's `base`, and importing it at the top would make the tool
        # registry depend on the loop that calls it.
        from ..agent import Agent, NullListener
        from ..context import file_tree
        from ..prompts import system_prompt
        from ..session import Session
        from ..skills import render as render_skills
        from . import build_registry

        try:
            tools, workspace = build_registry(root, read_only=True, skills=skills)
        except Exception as exc:
            raise ToolError(f"task: could not prepare a sub-agent: {exc}") from None

        session = Session(
            system=system_prompt(
                tools,
                str(workspace.root),
                file_tree(workspace.root),
                "",
                render_skills(skills) if skills else "",
            ),
            cwd=str(workspace.root),
            model=getattr(provider, "model", ""),
        )
        # Not persisted. The sub-agent's transcript is a search nobody asked to
        # keep, and writing one file per delegated question would bury the
        # sessions the user actually had in a list they have to scroll.
        sub = Agent(
            provider=provider,
            registry=tools,
            session=session,
            listener=Quiet(listener) if listener is not None else NullListener(),
            # No permissions object: the registry is read-only, so there is
            # nothing left to ask about, and a prompt raised from inside a tool
            # call would be a question the user has no context for.
            permissions=None,
            max_iterations=iterations,
            max_seconds=seconds,
            scout_root=workspace.root,
            # The parent's hooks, on purpose. A `pre_tool` gate keeping the
            # agent out of a directory is not a gate if delegating the read is
            # the way around it. The sub-agent's `turn_end` fires too, and
            # that is the honest reading: its turn did end.
            hooks=hooks,
        )

        outcome = sub.run(instruction)
        if not outcome.answer.strip():
            # Said as a fact, not an error: the parent asked a question and got
            # nothing, and it needs to know that so it can go and look itself
            # rather than write up an answer it never received.
            raise ToolError(
                f"task: the sub-agent stopped ({outcome.stopped}) without an "
                "answer. Do this part yourself, or ask something narrower."
            )

        return ToolResult.success(outcome.answer.strip())

    registry.add(
        Tool(
            name="task",
            description=DESCRIPTION,
            parameters=SCHEMA,
            run=run,
            # Reading only, so nothing to approve. The sub-agent's own tools
            # are the read-only set for exactly this reason.
            mutating=False,
        )
    )
