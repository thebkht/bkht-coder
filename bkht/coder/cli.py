"""Command line entry point."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

from . import banner, lineedit, markdown, narrate, terminal
from .agent import Agent
from .approval import ask_tty
from . import doctor
from .doctor import running_from, version
from .context import file_tree
from .instructions import load_instructions, render, summarize as summarize_instructions
from .parsing import ToolCall
from .permissions import ASK, AUTO, PLAN, Permissions, cycle as next_mode
from .prompts import system_prompt
from .prompt import Reader
from .provider import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    OllamaProvider,
)
from .repl import Repl
from .review import cli as review_cli
from .session import Session, Snapshots
from . import sessions as saved
from .skills import Discovery, discover as discover_skills, render as render_skills
from .skills import summarize as summarize_skills
from .status import Status
from .streaming import Gate
from .terminal import ACCENT, BOLD, DIM, GREEN, ORANGE, RED, YELLOW, paint
from . import usage
from .tools import build_registry
from .tools.background import Jobs
from .tools.base import ToolResult, set_output_budget


def summarize(arguments: dict) -> str:
    """A one-line rendering of tool arguments for the activity line.

    A call with a single argument drops the name: nobody reading
    ``read_file(bkht/coder/cli.py)`` wonders which argument that was, and the
    characters it saves are more of the path that fits on the line.
    """
    parts = []
    for key, value in arguments.items():
        text = str(value).replace("\n", "\\n")
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(text if len(arguments) == 1 else f"{key}={text}")
    return ", ".join(parts)


#: The status mark beside a tool call. Green once it worked, red when it did
#: not -- and never drawn before that is known, because a line which has
#: already scrolled cannot be repainted.
DOT = "\u25cf"


def duration(seconds: float) -> str:
    """A turn's length, in whole seconds once there is a whole second of it."""
    return f"{seconds:.1f}s" if seconds < 1 else f"{round(seconds)}s"


def cost(outcome) -> str:
    """What the turn took: how long it ran, and how many tokens went each way."""
    return f"{duration(outcome.seconds)} (\u2191{outcome.sent:,} \u2193{outcome.received:,})"


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
        # provider -> gate (drops tool-call JSON) -> markdown -> screen. The
        # gate comes first because what it removes was never prose, and a
        # half-written tool call passed through a renderer would be formatted.
        self.markdown = markdown.Stream(self._emit, colour=terminal.colourful(self.stream))
        self.gate = Gate(self.markdown.feed) if self.rich else None
        self.streamed = False
        self._streaming = False
        self._blocks = 0

    # --- turn boundary ------------------------------------------------------

    @contextlib.contextmanager
    def turn(self):
        """Bracket one agent.run(), so the spinner and the gate are bounded."""
        self.streamed = False
        self._blocks = 0
        self.status.start("thinking")
        try:
            yield self
        finally:
            self.status.stop()
            if self.gate is not None:
                # In that order: the gate releases what it held into the
                # renderer, and only then does the renderer flush its own tail.
                self.gate.finish()
                self.markdown.finish()
            self._newline()

    def footer(self, outcome) -> None:
        """The line under an answer saying what the turn cost.

        Outside :meth:`turn` because only the caller has the outcome, and a
        context manager that had to be handed one would be a worse shape than
        a second call.
        """
        if not self.rich or not outcome.answer:
            return
        self._gap()
        self._say(paint(cost(outcome), DIM, self.stream))

    def _gap(self) -> None:
        """A blank line above the next block, so the turn has a rhythm.

        Grouping is done by proximity here, and everything in a turn arriving
        flush against everything else leaves nothing for the eye to group by:
        the question, each call, the answer and its cost are four things, not
        one paragraph.

        Chrome, so it stops at the edge of a terminal. A pipe gets the lines it
        always got, because something downstream is parsing them.
        """
        if not self.rich:
            return
        self._blocks += 1
        with self.status.pause():
            self._newline()
            print("", file=self.stream, flush=True)

    def _emit(self, text: str) -> None:
        # The first prose of a turn opens a block of its own, whether or not a
        # tool call came before it.
        if not self.streamed:
            self._gap()
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
        # Nothing is printed yet. The call's line is drawn when its result is
        # known, so the mark beside it can say which way it went; until then
        # the status line is what says the call is running, and it says it in
        # the same words the printed sentence used to.
        self.status.note(narrate.intent(call))

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        self._gap()
        mark = paint(DOT, GREEN if result.ok else RED, self.stream)
        name = paint(call.name, ORANGE, self.stream)
        self._say(f"{mark} {name}({summarize(call.arguments)})")
        if not result.ok:
            self._say(paint(f"  ! {result.error.splitlines()[0]}", RED, self.stream))
        elif self.verbose:
            for line in result.content.splitlines()[:20]:
                self._say(paint(f"  {line}", DIM, self.stream))
        self.status.note("thinking")

    def on_retry(self, reason: str) -> None:
        """Report why the loop is doing something other than answering.

        The reason is printed rather than discarded. It used to be replaced with
        a fixed "retrying (malformed reply)", which was wrong for the caller that
        fires most often -- freeing context -- and sent a real diagnosis off in
        the wrong direction for two turns.
        """
        self._gap()
        self._say(paint(f"{DOT} {reason}", YELLOW, self.stream))


def add_common_arguments(parser) -> None:
    """Flags shared by the agent and by `coder review`."""
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL.")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Context window to request.")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature. Low keeps tool calls well-formed.",
    )
    parser.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")
    parser.add_argument("--auto", action="store_true", help="Allow every tool call without prompting.")
    parser.add_argument("--no-instructions", action="store_true", help="Ignore AGENTS.md and CLAUDE.md.")
    parser.add_argument("--no-skills", action="store_true", help="Ignore skills, and omit the skill tool.")


def add_agent_arguments(parser) -> None:
    """Flags for running a task, as opposed to a subcommand."""
    parser.add_argument("prompt", nargs="*", help="Task to run. Omit for an interactive session.")
    # An optional value rather than a switch: `--resume` still means "the newest
    # session here", and `--resume <id>` reaches any of the others.
    parser.add_argument(
        "--resume", nargs="?", const=saved.LAST, default=None, metavar="last|ID",
        help="Continue a saved session. Defaults to the newest for this directory.",
    )
    parser.add_argument("--plan", action="store_true", help="Read-only: refuse every change to the workspace.")
    parser.add_argument("--verbose", action="store_true", help="Stream raw model output and tool results.")
    parser.add_argument("--no-scout", action="store_true", help="Do not search the workspace before each task.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Cap on loop iterations per task.")
    add_common_arguments(parser)


def version_line() -> str:
    """What `--version` prints: the version, and which copy printed it.

    The path is not decoration. A `uv tool install` puts a second copy of coder
    on PATH, and when the two disagree the only question worth answering first
    is which one just ran.
    """
    return " ".join(filter(None, ("coder", version(), f"({running_from()})")))


def build_parser() -> argparse.ArgumentParser:
    parser = usage.Parser(
        prog="coder",
        description="A coding agent running against a local Ollama server.",
        page=usage.HELP,
    )
    parser.add_argument("-v", "--version", action="version", version=version_line())
    subparsers = parser.add_subparsers(dest="command", parser_class=usage.Parser)
    reviewer = subparsers.add_parser(
        "review", help="Review uncommitted changes, a branch, or a commit range.",
        page=usage.REVIEW_HELP,
    )
    add_common_arguments(reviewer)
    review_cli.add_arguments(reviewer)

    checker = subparsers.add_parser(
        "doctor", help="Check that this install can actually run a turn.",
        page=usage.DOCTOR_HELP,
    )
    add_common_arguments(checker)
    doctor.add_arguments(checker)

    lister = subparsers.add_parser(
        "sessions", help="List saved sessions for this workspace.",
        page=usage.SESSIONS_HELP,
    )
    saved_cli_arguments(lister)

    opener = subparsers.add_parser(
        "session", help="Show or resume one saved session.",
        page=usage.SESSION_HELP,
    )
    opener.add_argument("target", nargs="?", default=saved.LAST, help="`last`, or a session id.")
    opener.add_argument("--json", action="store_true", help="Emit the session as JSON.")
    opener.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")

    add_agent_arguments(parser)
    return parser


def saved_cli_arguments(parser) -> None:
    """Flags for `coder sessions`, which needs a workspace and nothing else."""
    parser.add_argument("--all", action="store_true", help="Every workspace, not just this one.")
    parser.add_argument("--json", action="store_true", help="Emit the listing as JSON.")
    parser.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")


def build_agent_parser() -> argparse.ArgumentParser:
    """The parser used when the first argument is not a subcommand.

    A parser carrying both a subcommand and a free-text positional cannot tell
    them apart: argparse matches the first positional against the subcommand
    list, so `coder "add a --verbose flag"` failed as an invalid choice. The
    two are therefore chosen between before parsing, not during it.
    """
    parser = usage.Parser(
        prog="coder",
        description="A coding agent running against a local Ollama server.",
        page=usage.HELP,
    )
    parser.add_argument("-v", "--version", action="version", version=version_line())
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


class Loaded(NamedTuple):
    """What shaped this session before the first turn: instructions, and skills.

    One line each, already summarised, and empty when there was nothing to say.
    The greeting sets them inside the box; a one-shot run prints them above its
    answer, which is where they have always been.
    """

    instructions: str
    skills: str

    def lines(self) -> list[str]:
        """One entry per line: a skipped skill is reported on its own row."""
        return [
            line
            for summary in (self.instructions, self.skills)
            for line in summary.splitlines()
            if line
        ]


def make_agent(args, listener=None) -> tuple[Agent, Snapshots]:
    """Wire up provider, tools, permissions, session, jobs, and agent."""
    root = Path(args.cwd).expanduser().resolve()
    mode = resolve_mode(args)
    snapshots = Snapshots()
    jobs = Jobs()
    # Registered before anything can start a process, and unconditionally: a
    # session that dies on an exception must not leave a server behind either.
    atexit.register(jobs.stop_all)

    # Discovered before the registry is built: whether the `skill` tool exists
    # at all depends on whether there is anything for it to fetch.
    found_skills = (
        Discovery() if getattr(args, "no_skills", False) else discover_skills(root)
    )

    # In plan mode the mutating tools are left out of the registry entirely
    # rather than denied at call time, so the model is never tempted by a tool
    # it cannot use -- one fewer way for a small model to waste a turn.
    registry, workspace = build_registry(
        root, read_only=(mode == PLAN), snapshots=snapshots, skills=found_skills,
        jobs=jobs,
    )
    # The prompt pauses the status line for the whole exchange: without it the
    # spinner repaints over the diff being approved.
    pause = listener.status.pause if getattr(listener, "status", None) else None
    permissions = Permissions(
        mode=mode, workspace=workspace,
        **({"prompt": partial(ask_tty, pause=pause)} if terminal.interactive() else {}),
    )
    # A permissions file that could not be read is announced, never swallowed:
    # a session that believes it has rules and does not is the one case where
    # the user would be surprised by a prompt they thought they had answered.
    if permissions.rules is not None and permissions.rules.error:
        print(paint(permissions.rules.error, YELLOW, sys.stderr), file=sys.stderr)
    provider = OllamaProvider(
        model=args.model, host=args.host, num_ctx=args.num_ctx,
        temperature=args.temperature,
    )
    # Tool output is capped as a share of the window, so the window has to be
    # known before any tool runs.
    set_output_budget(provider.num_ctx)

    # Announced rather than applied silently: instructions shape every answer
    # the model gives, and a rule the user has forgotten writing is worse than
    # no rule at all.
    loaded = [] if getattr(args, "no_instructions", False) else load_instructions(workspace.root)
    # Skills are announced for the same reason instructions are: a rule that
    # shapes answers silently is worse than no rule. Skipped ones are named
    # too -- a skill that never loads looks exactly like one the model chose
    # not to use.
    #
    # Returned rather than printed here: a session that draws the greeting has
    # somewhere better to put these two lines than above it.
    announced = Loaded(
        summarize_instructions(loaded) if loaded else "",
        summarize_skills(found_skills) if found_skills.skills or found_skills.problems else "",
    )
    system = system_prompt(
        registry,
        str(workspace.root),
        file_tree(workspace.root),
        render(loaded),
        render_skills(found_skills),
    )

    # The system prompt is rebuilt rather than reloaded, so a resumed session
    # picks up the current tool set instead of whatever it was told last time.
    session = None
    target = getattr(args, "resume", None)
    if target:
        previous = saved.resolve(workspace.root, target)
        if previous is None:
            # Named ids and `last` fail differently: one is a directory with no
            # history yet, the other is a typo, and starting fresh is only the
            # obvious thing to do about the first.
            print(paint(f"{saved.missing(workspace.root, target)} Starting a new one.", YELLOW, sys.stderr), file=sys.stderr)
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
    return agent, snapshots, permissions, workspace, jobs, announced


def report(outcome, streamed: bool = False) -> int:
    """Print an outcome and return the process exit status.

    ``streamed`` means the listener already put this prose on the screen as it
    arrived, so printing it again would show the answer twice.
    """
    if outcome.answer and not streamed:
        # Rendered here too. A piped run and a --verbose one never reached the
        # listener's renderer, and an answer full of raw asterisks is no more
        # readable for having been redirected.
        # rstrip, because render() ends every document with a newline and
        # print would add a second one.
        print(markdown.render(outcome.answer, colour=terminal.colourful()).rstrip("\n"))

    if outcome.stopped != "answered":
        # Only when there is nothing else to show. The last error is a message
        # written for the model -- "you already ran read_file with exactly these
        # arguments" -- and printing it beneath a real answer reads as though it
        # were the answer's caveat.
        detail = "" if outcome.answer else (outcome.errors[-1] if outcome.errors else "")
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
    listener.footer(outcome)
    return report(outcome, streamed=listener.streamed)


def parts(agent, permissions) -> tuple[str, str, str]:
    """Model, mode and context: the three things that change what a turn does.

    Separately, because the box stacks them in a column half a screen wide and
    every other caller wants them on one line.
    """
    num_ctx = getattr(agent.provider, "num_ctx", DEFAULT_NUM_CTX)
    used = agent.session.prompt_tokens
    context = f"{used:,}/{num_ctx:,} ctx" if num_ctx else f"{used:,} tokens"
    return agent.provider.model, permissions.mode, context


def facts(agent, permissions) -> str:
    """The three of them on one line."""
    return " · ".join(parts(agent, permissions))


def header(agent, permissions, workspace) -> str:
    return f"coder · {facts(agent, permissions)} · {workspace.root}"


HINT = "/help for commands, /exit to leave."

#: The prompt itself. A single mark, so the input starts as near the left edge
#: as it can: what is being typed is the only thing on that row worth reading.
PROMPT = "› "


def home_relative(path) -> str:
    """``~/project`` where that is the same place, for a column with an edge."""
    text = str(path)
    try:
        return "~/" + str(Path(text).relative_to(Path.home()))
    except ValueError:
        return text


def greeting(agent, permissions, workspace, stream=None, loaded=None) -> str:
    """What the session opens with: the mark, and four facts beside it.

    The banner is chrome, and chrome that reaches a pipe is noise -- worse,
    noise that something downstream is already parsing. Off a terminal, in a
    window too narrow to hold the art, or in a locale that cannot draw it, this
    is exactly the two lines it has always been.

    One shape, not three. What the greeting has to say is a name, a model and a
    place, and that fits beside the art at any width worth drawing art at; a
    second, wider layout was a second thing to keep true of the first.
    """
    stream = sys.stdout if stream is None else stream
    plain = f"{header(agent, permissions, workspace)}\n{HINT}"
    if not terminal.interactive(stream):
        return plain
    if terminal.width() < banner.MIN_WIDTH or not banner.drawable(stream):
        return paint(plain, DIM, stream)

    _, mode, context = parts(agent, permissions)
    # Only the name is coloured. What a greeting is for is being read once and
    # then ignored, and a column of accents is a column that keeps asking.
    return banner.render([
        paint(" ".join(filter(None, ("bkht.coder", version()))), ACCENT, stream),
        paint(agent.provider.model, DIM, stream),
        paint(home_relative(workspace.root), DIM, stream),
        paint(f"{mode} · {context}", DIM, stream),
        *(paint(line, DIM, stream) for line in (loaded.lines() if loaded else [])),
    ])


def divider() -> str:
    """A blank line and a rule, or nothing at all off a terminal.

    Drawn above each prompt so one exchange ends where the next begins: a long
    turn scrolled back through is otherwise an unbroken wall, with no line to
    say where the answer to the last question started. The blank line above it
    is what makes it a boundary rather than the last line of the answer.
    """
    if not terminal.interactive():
        return ""
    return f"\n{paint(banner.rule(terminal.width()), DIM)}"


def read_line(reader, prompt: str, stream=None) -> str:
    """Read one line, however this reader draws one.

    A reader with its own editor draws its own frame -- rules above and below
    the input, and a footer naming the permission mode -- and is left to it.
    What follows is for the readline and ``input()`` paths, which cannot.

    The hint is parked on the row beneath the input.

    Under the input is where it belongs -- it is about what you are typing, and
    proximity is what says so. Under the input is also where the cursor is, so
    the row is written first and the cursor walked back up over it; readline
    then edits the row above a line that is already on screen.

    Enter leaves the cursor back on the hint's row, so it is cleared there. The
    hint is for the moment you are typing; left behind, it would sit stranded
    in the scrollback above an answer it says nothing about.
    """
    if getattr(reader, "cycles", False):
        return reader.read(prompt)

    stream = sys.stdout if stream is None else stream
    if not terminal.interactive(stream):
        return reader.read(paint(prompt, ACCENT + BOLD, stream))

    stream.write(f"\n{paint(HINT, DIM, stream)}{terminal.CURSOR_UP}\r")
    stream.flush()
    try:
        return reader.read(paint(prompt, ACCENT + BOLD, stream))
    finally:
        stream.write(terminal.CLEAR_LINE)
        stream.flush()


def interactive(agent, snapshots, permissions, workspace, listener,
                use_instructions=True, jobs=None, loaded=None) -> int:
    """The REPL. Ctrl-C abandons the current line; Ctrl-D leaves."""
    repl = Repl(
        agent, snapshots, permissions, workspace,
        use_instructions=use_instructions, jobs=jobs,
    )
    # The footer is a callable, not a string: the mode it names is the thing
    # Shift+Tab changes, and the editor redraws it on the same keypress.
    reader = Reader(
        repl,
        enabled=terminal.interactive(),
        footer=lambda: lineedit.footer(permissions.mode),
        cycle=lambda: setattr(permissions, "mode", next_mode(permissions.mode)),
    )
    # Printed as it comes back: the greeting dims and bolds its own parts now,
    # and a paint() around the whole of it would flatten both.
    print(greeting(agent, permissions, workspace, loaded=loaded))

    while True:
        try:
            # The editor opens with a rule of its own; a divider above it
            # would be the same line drawn twice.
            if rule := (divider() if not reader.cycles else ""):
                print(rule)
            elif reader.cycles:
                print()
            line = read_line(reader, PROMPT)
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


def resume_argv(rest: list[str]) -> list[str]:
    """`session resume [id] [flags]` as the agent parser's own arguments.

    A bare `resume`, or one whose next word is a flag, means the newest session
    for this directory -- the same thing `--resume` has always meant.
    """
    named = bool(rest[:1]) and not rest[0].startswith("-")
    return ["--resume", rest[0] if named else saved.LAST, *rest[1 if named else 0:]]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv[:1] == ["review"]:
        return review_cli.run(build_parser().parse_args(argv))

    if argv[:1] == ["help"] or argv[:1] == ["--help"] or argv[:1] == ["-h"]:
        print(usage.HELP)
        return 0

    if argv[:1] == ["sessions"]:
        args = build_parser().parse_args(argv)
        return saved.report(
            Path(args.cwd).expanduser().resolve(),
            as_json=args.json, everywhere=args.all,
        )

    if argv[:1] == ["session"]:
        # `resume` re-enters the ordinary agent path rather than reimplementing
        # it, so a resumed session gets the current tools, instructions and
        # permission mode like any other.
        rest = argv[1:]
        if rest[:1] == ["resume"]:
            argv = resume_argv(rest[1:])
        else:
            args = build_parser().parse_args(argv)
            return saved.show(
                Path(args.cwd).expanduser().resolve(),
                args.target, as_json=args.json,
            )

    if argv[:1] == ["doctor"]:
        args = build_parser().parse_args(argv)
        return doctor.report(
            Path(args.cwd).expanduser().resolve(),
            model=args.model, host=args.host, num_ctx=args.num_ctx,
            as_json=args.json,
        )

    args = build_agent_parser().parse_args(argv)
    listener = TerminalListener(verbose=args.verbose)
    agent, snapshots, permissions, workspace, jobs, announced = make_agent(args, listener)

    try:
        if args.prompt:
            # A one-shot run draws no greeting, so what loaded is announced the
            # way it always was: above the answer it shaped.
            for line in announced.lines():
                print(paint(line, DIM))
            return run_turn(agent, listener, " ".join(args.prompt))
        return interactive(
            agent, snapshots, permissions, workspace, listener,
            use_instructions=not args.no_instructions, jobs=jobs, loaded=announced,
        )
    finally:
        # The user started nothing here and can see nothing here: a server left
        # running after the session ends is one they have no way to find again.
        jobs.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
