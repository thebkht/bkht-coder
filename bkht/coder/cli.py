"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .context import file_tree
from .parsing import ToolCall
from .permissions import ASK, AUTO, PLAN, Permissions
from .prompts import system_prompt
from .provider import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_NUM_CTX, OllamaProvider
from .repl import Repl
from .session import Session, Snapshots
from .tools import build_registry
from .tools.base import ToolResult

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def paint(text: str, colour: str, stream=None) -> str:
    """Colour ``text`` only when the destination is a terminal."""
    stream = stream or sys.stdout
    return f"{colour}{text}{RESET}" if stream.isatty() else text


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
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._streaming = False

    def on_token(self, text: str) -> None:
        # Tool-call JSON is streamed too; showing it would be noise.
        if not self.verbose:
            return
        self._streaming = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def _newline(self) -> None:
        if self._streaming:
            sys.stdout.write("\n")
            self._streaming = False

    def on_tool_call(self, call: ToolCall) -> None:
        self._newline()
        label = f"{call.name}({summarize(call.arguments)})"
        print(paint(f"  · {label}", CYAN))

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        if not result.ok:
            print(paint(f"    ! {result.error.splitlines()[0]}", RED))
        elif self.verbose:
            for line in result.content.splitlines()[:20]:
                print(paint(f"    {line}", DIM))

    def on_retry(self, reason: str) -> None:
        self._newline()
        print(paint("  · retrying (malformed reply)", YELLOW))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coder",
        description="A coding agent running against a local Ollama server.",
    )
    parser.add_argument("prompt", nargs="*", help="Task to run. Omit for an interactive session.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL.")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Context window to request.")
    parser.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")
    parser.add_argument("--resume", action="store_true", help="Continue the most recent session for this directory.")
    parser.add_argument("--auto", action="store_true", help="Allow every tool call without prompting.")
    parser.add_argument("--plan", action="store_true", help="Read-only: refuse every change to the workspace.")
    parser.add_argument("--verbose", action="store_true", help="Stream raw model output and tool results.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Cap on loop iterations per task.")
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
    permissions = Permissions(mode=mode, workspace=workspace)
    provider = OllamaProvider(model=args.model, host=args.host, num_ctx=args.num_ctx)
    system = system_prompt(registry, str(workspace.root), file_tree(workspace.root))

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
    )
    return agent, snapshots, permissions, workspace


def report(outcome) -> int:
    """Print an outcome and return the process exit status."""
    if outcome.answer:
        print(outcome.answer)

    if outcome.stopped != "answered":
        detail = outcome.errors[-1] if outcome.errors else ""
        print(
            paint(f"[stopped: {outcome.stopped}] {detail}".strip(), YELLOW, sys.stderr),
            file=sys.stderr,
        )
        return 1
    return 0


def interactive(agent, snapshots, permissions, workspace, listener) -> int:
    """The REPL. Ctrl-C abandons the current line; Ctrl-D leaves."""
    repl = Repl(agent, snapshots, permissions, workspace)
    print(paint(f"coder · {agent.provider.model} · {permissions.mode} · {workspace.root}", DIM))
    print(paint("/help for commands, /exit to leave.", DIM))

    while True:
        try:
            line = input(paint("> ", BOLD))
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
            report(agent.run(command.task))
        except KeyboardInterrupt:
            # Abandon this task but keep the session; a long local turn is
            # exactly the thing a user needs to be able to interrupt.
            print(paint("\n[interrupted]", YELLOW))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    listener = TerminalListener(verbose=args.verbose)
    agent, snapshots, permissions, workspace = make_agent(args, listener)

    if args.prompt:
        return report(agent.run(" ".join(args.prompt)))
    return interactive(agent, snapshots, permissions, workspace, listener)


if __name__ == "__main__":
    raise SystemExit(main())
