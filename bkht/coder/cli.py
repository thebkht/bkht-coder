"""Command line entry point."""

from __future__ import annotations

import argparse
import contextlib
import sys
from functools import partial
from pathlib import Path

from . import narrate, terminal
from .agent import Agent
from .approval import ask_tty
from .context import file_tree
from .instructions import load_instructions, render, summarize as summarize_instructions
from .parsing import ToolCall
from .permissions import ASK, AUTO, PLAN, Permissions
from .prompts import system_prompt
from .prompt import Reader
from .provider import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_NUM_CTX, OllamaProvider
from .repl import Repl
from .review import cli as review_cli
from .session import Session, Snapshots
from .status import Status
from .streaming import Gate
from .terminal import BOLD, CYAN, DIM, RED, YELLOW, paint
from .tools import build_registry
from .tools.base import ToolResult


def summarize(arguments: dict) -> str:
    """A one-line rendering of tool arguments for the activity line."""
    parts = []
    for key, value in arguments.items():
        text = str(value).replace("\n", "\\n")
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


class TerminalListener:
    """Renders loop events as they happen.

    Streaming output matters more here than it does against a hosted model:
    waiting 30 seconds with no sign of life is the main thing that makes a
    local agent feel dead.

    On a terminal the prose streams as the model writes it, with tool-call JSON
    held back by the gate and a status line filling the silences. A pipe or a
    file gets none of that -- no colour, no spinner, no streamed tokens -- just
    the narrated tool lines and the finished answer.
    """

    def __init__(self, verbose: bool = False, live: bool | None = None, stream=None) -> None:
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.live = terminal.interactive() if live is None else live
        # Not while --verbose: raw output is continuous, so there is no silence
        # for a spinner to fill and nothing to gate.
        self.rich = self.live and not verbose
        self.status = Status(writer=self.stream, enabled=self.rich)
        self.gate = Gate(self._emit) if self.rich else None
        self.streamed = False
        self._streaming = False

    # --- turn boundary ------------------------------------------------------

    @contextlib.contextmanager
    def turn(self):
        """Bracket one agent.run(), so the spinner and the gate are bounded."""
        self.streamed = False
        self.status.start("thinking")
        try:
            yield self
        finally:
            self.status.stop()
            if self.gate is not None:
                self.gate.finish()
            self._newline()

    def _emit(self, text: str) -> None:
        with self.status.pause():
            self.stream.write(text)
            self.stream.flush()
            # Set inside the pause, so its redraw already knows to hold off.
            self.status.suspend(not text.endswith("\n"))
        self.streamed = True
        self._streaming = not text.endswith("\n")

    # --- loop events --------------------------------------------------------

    def on_token(self, text: str) -> None:
        if self.gate is not None:
            self.status.add_tokens()
            self.gate.feed(text)
            return
        # Tool-call JSON is streamed too; showing it would be noise.
        if not self.verbose:
            return
        self._streaming = True
        self.stream.write(text)
        self.stream.flush()

    def _newline(self) -> None:
        if self._streaming:
            self.stream.write("\n")
            self.stream.flush()
            self._streaming = False
        self.status.suspend(False)

    def _say(self, text: str) -> None:
        with self.status.pause():
            self._newline()
            print(text, file=self.stream, flush=True)

    def on_tool_call(self, call: ToolCall) -> None:
        # The sentence is what gets read; the call underneath is the evidence
        # for it, which is why both are printed and neither replaces the other.
        self._say(
            paint(f"  · {narrate.intent(call)}", CYAN, self.stream)
            + "\n"
            + paint(f"    {call.name}({summarize(call.arguments)})", DIM, self.stream)
        )
        self.status.note(narrate.label(call))

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        if not result.ok:
            self._say(paint(f"    ! {result.error.splitlines()[0]}", RED, self.stream))
        elif self.verbose:
            for line in result.content.splitlines()[:20]:
                self._say(paint(f"    {line}", DIM, self.stream))
        self.status.note("thinking")

    def on_retry(self, reason: str) -> None:
        self._say(paint("  · retrying (malformed reply)", YELLOW, self.stream))


def add_common_arguments(parser) -> None:
    """Flags shared by the agent and by `coder review`."""
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL.")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Context window to request.")
    parser.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")
    parser.add_argument("--auto", action="store_true", help="Allow every tool call without prompting.")
    parser.add_argument("--no-instructions", action="store_true", help="Ignore AGENTS.md and CLAUDE.md.")


def add_agent_arguments(parser) -> None:
    """Flags for running a task, as opposed to a subcommand."""
    parser.add_argument("prompt", nargs="*", help="Task to run. Omit for an interactive session.")
    parser.add_argument("--resume", action="store_true", help="Continue the most recent session for this directory.")
    parser.add_argument("--plan", action="store_true", help="Read-only: refuse every change to the workspace.")
    parser.add_argument("--verbose", action="store_true", help="Stream raw model output and tool results.")
    parser.add_argument("--no-scout", action="store_true", help="Do not search the workspace before each task.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Cap on loop iterations per task.")
    add_common_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coder",
        description="A coding agent running against a local Ollama server.",
    )
    subparsers = parser.add_subparsers(dest="command")
    reviewer = subparsers.add_parser(
        "review", help="Review uncommitted changes, a branch, or a commit range."
    )
    add_common_arguments(reviewer)
    review_cli.add_arguments(reviewer)
    add_agent_arguments(parser)
    return parser


def build_agent_parser() -> argparse.ArgumentParser:
    """The parser used when the first argument is not a subcommand.

    A parser carrying both a subcommand and a free-text positional cannot tell
    them apart: argparse matches the first positional against the subcommand
    list, so `coder "add a --verbose flag"` failed as an invalid choice. The
    two are therefore chosen between before parsing, not during it.
    """
    parser = argparse.ArgumentParser(
        prog="coder",
        description="A coding agent running against a local Ollama server.",
        epilog="Run `coder review --help` for the code-review subcommand.",
    )
    add_agent_arguments(parser)
    return parser


def resolve_mode(args) -> str:
    if args.auto and args.plan:
        raise SystemExit("--auto and --plan contradict each other; pick one.")
    if args.auto:
        return AUTO
    if args.plan:
        return PLAN
    return ASK


def make_agent(args, listener=None) -> tuple[Agent, Snapshots]:
    """Wire up provider, tools, permissions, session, and agent."""
    root = Path(args.cwd).expanduser().resolve()
    mode = resolve_mode(args)
    snapshots = Snapshots()

    # In plan mode the mutating tools are left out of the registry entirely
    # rather than denied at call time, so the model is never tempted by a tool
    # it cannot use -- one fewer way for a small model to waste a turn.
    registry, workspace = build_registry(
        root, read_only=(mode == PLAN), snapshots=snapshots
    )
    # The prompt pauses the status line for the whole exchange: without it the
    # spinner repaints over the diff being approved.
    pause = listener.status.pause if getattr(listener, "status", None) else None
    permissions = Permissions(
        mode=mode, workspace=workspace,
        **({"prompt": partial(ask_tty, pause=pause)} if terminal.interactive() else {}),
    )
    provider = OllamaProvider(model=args.model, host=args.host, num_ctx=args.num_ctx)

    # Announced rather than applied silently: instructions shape every answer
    # the model gives, and a rule the user has forgotten writing is worse than
    # no rule at all.
    loaded = [] if getattr(args, "no_instructions", False) else load_instructions(workspace.root)
    if loaded:
        print(paint(summarize_instructions(loaded), DIM))
    system = system_prompt(
        registry, str(workspace.root), file_tree(workspace.root), render(loaded)
    )

    # The system prompt is rebuilt rather than reloaded, so a resumed session
    # picks up the current tool set instead of whatever it was told last time.
    session = None
    if getattr(args, "resume", False):
        previous = Session.latest_for(str(workspace.root))
        if previous is None:
            print(paint("No previous session for this directory; starting a new one.", YELLOW, sys.stderr), file=sys.stderr)
        else:
            session = Session.load(previous, system=system)
            print(paint(f"Resumed {previous.name} ({len(session.messages)} messages).", DIM))

    if session is None:
        session = Session(system=system, cwd=str(workspace.root), model=args.model)
        session.start_file()

    agent = Agent(
        provider=provider,
        registry=registry,
        session=session,
        listener=listener,
        permissions=permissions,
        max_iterations=args.max_iterations,
        scout_root=None if getattr(args, "no_scout", False) else workspace.root,
    )
    return agent, snapshots, permissions, workspace


def report(outcome, streamed: bool = False) -> int:
    """Print an outcome and return the process exit status.

    ``streamed`` means the listener already put this prose on the screen as it
    arrived, so printing it again would show the answer twice.
    """
    if outcome.answer and not streamed:
        print(outcome.answer)

    if outcome.stopped != "answered":
        detail = outcome.errors[-1] if outcome.errors else ""
        print(
            paint(f"[stopped: {outcome.stopped}] {detail}".strip(), YELLOW, sys.stderr),
            file=sys.stderr,
        )
        return 1
    return 0


def run_turn(agent, listener, task: str) -> int:
    """One task, with the spinner and the gate bounded to it."""
    if listener is None:
        return report(agent.run(task))
    with listener.turn():
        outcome = agent.run(task)
    return report(outcome, streamed=listener.streamed)


def header(agent, permissions, workspace) -> str:
    num_ctx = getattr(agent.provider, "num_ctx", DEFAULT_NUM_CTX)
    used = agent.session.prompt_tokens
    context = f"{used:,}/{num_ctx:,} ctx" if num_ctx else f"{used:,} tokens"
    return (
        f"coder · {agent.provider.model} · {permissions.mode} · "
        f"{context} · {workspace.root}"
    )


def interactive(agent, snapshots, permissions, workspace, listener, use_instructions=True) -> int:
    """The REPL. Ctrl-C abandons the current line; Ctrl-D leaves."""
    repl = Repl(agent, snapshots, permissions, workspace, use_instructions=use_instructions)
    reader = Reader(repl, enabled=terminal.interactive())
    print(paint(header(agent, permissions, workspace), DIM))
    print(paint("/help for commands, /exit to leave.", DIM))

    while True:
        try:
            line = reader.read(paint("> ", BOLD))
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        command = repl.dispatch(line)
        if command.quit:
            return 0
        if command.handled:
            continue

        try:
            run_turn(agent, listener, command.task)
        except KeyboardInterrupt:
            # Abandon this task but keep the session; a long local turn is
            # exactly the thing a user needs to be able to interrupt.
            print(paint("\n[interrupted]", YELLOW))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv[:1] == ["review"]:
        return review_cli.run(build_parser().parse_args(argv))

    args = build_agent_parser().parse_args(argv)
    listener = TerminalListener(verbose=args.verbose)
    agent, snapshots, permissions, workspace = make_agent(args, listener)

    if args.prompt:
        return run_turn(agent, listener, " ".join(args.prompt))
    return interactive(
        agent, snapshots, permissions, workspace, listener,
        use_instructions=not args.no_instructions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
