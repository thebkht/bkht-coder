"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .parsing import ToolCall
from .prompts import system_prompt
from .provider import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_NUM_CTX, OllamaProvider
from .session import Session
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
    parser.add_argument("--verbose", action="store_true", help="Stream raw model output and tool results.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Cap on loop iterations per task.")
    return parser


def make_agent(args, listener=None) -> Agent:
    """Wire up provider, tools, session, and agent from parsed arguments."""
    root = Path(args.cwd).expanduser().resolve()
    registry, workspace = build_registry(root, read_only=True)
    provider = OllamaProvider(model=args.model, host=args.host, num_ctx=args.num_ctx)
    session = Session(system=system_prompt(registry, str(workspace.root)))
    return Agent(
        provider=provider,
        registry=registry,
        session=session,
        listener=listener,
        max_iterations=args.max_iterations,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.prompt:
        print("Interactive mode is not built yet; pass a prompt.", file=sys.stderr)
        return 2

    agent = make_agent(args, TerminalListener(verbose=args.verbose))
    outcome = agent.run(" ".join(args.prompt))

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


if __name__ == "__main__":
    raise SystemExit(main())
