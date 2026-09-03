"""Command line entry point."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

from . import banner, cancel, clipboard, config, git, lineedit, markdown, narrate, terminal, update
from .agent import Agent
from .approval import ask_tty
from . import doctor
from .doctor import running_from, version
from .context import file_tree, usage_ratio
from .instructions import load_instructions, render, summarize as summarize_instructions
from .parsing import ToolCall
from .permissions import ASK, AUTO, PLAN, Permissions, cycle as next_mode
from .prompts import system_prompt
from .prompt import Reader
from .provider import BACKENDS, DEFAULT_NUM_CTX, DEFAULT_PROVIDER
from .provider import build as build_provider
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
        self._at = 0  # columns of prose on the line the cursor is on

    # --- turn boundary ------------------------------------------------------

    @contextlib.contextmanager
    def turn(self):
        """Bracket one agent.run(), so the spinner and the gate are bounded."""
        self.streamed = False
        self._blocks = 0
        self._at = 0
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
            # Set inside the pause, so its redraw already knows where the
            # sentence got to and draws itself below rather than over it.
            self.status.inline(self._column(text))
        self.streamed = True
        self._streaming = not text.endswith("\n")

    def _column(self, text: str) -> int:
        """How far along the line the prose just written reached.

        Counted in columns the terminal actually draws, so the colour a
        renderer wrapped a word in does not push the count past the word.

        Prose landing exactly on the right-hand edge is wrapped here rather
        than left to the terminal. A terminal in that position has not moved
        the cursor down yet -- it holds it against the last column with the
        wrap pending -- so there is no column to come back to that is not
        either the end of the line or the start of it. Writing the newline
        ourselves puts the cursor somewhere nameable, and the screen looks the
        same either way because that is where it would have wrapped.
        """
        if "\n" in text:
            self._at = 0 if text.endswith("\n") else terminal.visible(text.rsplit("\n", 1)[1])
        else:
            self._at += terminal.visible(text)
        if self._at and self._at % max(1, terminal.width()) == 0:
            self.stream.write("\n")
            self._at = 0
        return self._at

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
        self._at = 0
        self.status.inline(0)

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
        elif self.rich:
            # What came back, in a count. Without it the transcript showed the
            # call and never its result, so a model writing out a file it had
            # lost to compaction looked exactly like one that had read it.
            #
            # Chrome, so it stops at the edge of a terminal: a pipe gets the
            # tool lines it has always got, because something downstream is
            # parsing them.
            if said := narrate.outcome(call, result.content):
                self._say(paint(f"  {said}", DIM, self.stream))
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
    # Unset defaults to None rather than to the built-in value, so that
    # `config.Settings.apply` can tell a flag the user typed from one argparse
    # filled in -- without that distinction a config file could never win.
    parser.add_argument(
        "--provider", default=None, choices=sorted(BACKENDS),
        help="Model backend to run the turn through.",
    )
    parser.add_argument("--model", default=None, help="Model to use.")
    parser.add_argument("--host", default=None, help="Ollama server URL.")
    parser.add_argument("--num-ctx", type=int, default=None, help="Context window to request.")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Low keeps tool calls well-formed.",
    )
    parser.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")
    parser.add_argument("--auto", action="store_true", default=None, help="Allow every tool call without prompting.")
    parser.add_argument("--no-instructions", action="store_true", default=None, help="Ignore AGENTS.md and CLAUDE.md.")
    parser.add_argument("--no-skills", action="store_true", default=None, help="Ignore skills, and omit the skill tool.")


def add_agent_arguments(parser) -> None:
    """Flags for running a task, as opposed to a subcommand."""
    parser.add_argument("prompt", nargs="*", help="Task to run. Omit for an interactive session.")
    # An optional value rather than a switch: `--resume` still means "the newest
    # session here", and `--resume <id>` reaches any of the others.
    parser.add_argument(
        "--resume", nargs="?", const=saved.LAST, default=None, metavar="last|ID",
        help="Continue a saved session. Defaults to the newest for this directory.",
    )
    parser.add_argument("--plan", action="store_true", default=None, help="Read-only: refuse every change to the workspace.")
    parser.add_argument("--verbose", action="store_true", help="Stream raw model output and tool results.")
    parser.add_argument("--no-scout", action="store_true", default=None, help="Do not search the workspace before each task.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Cap on loop iterations per task.")
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

    updater = subparsers.add_parser(
        "update", help="Install the newest release, or check whether there is one.",
        page=usage.UPDATE_HELP,
    )
    updater.add_argument("--check", action="store_true", help="Report what is available; install nothing.")

    settings = subparsers.add_parser(
        "config", help="Show or change the settings that survive a restart.",
        page=usage.CONFIG_HELP,
    )
    add_config_arguments(settings)

    lister = subparsers.add_parser(
        "sessions", help="List saved sessions, this agent's or another's.",
        page=usage.SESSIONS_HELP,
    )
    saved_cli_arguments(lister)

    opener = subparsers.add_parser(
        "session", help="Show or resume one saved session.",
        page=usage.SESSION_HELP,
    )
    opener.add_argument("target", nargs="?", default=saved.LAST, help="`last`, a session id, or `claude/<id>`.")
    opener.add_argument("--json", action="store_true", help="Emit the session as JSON.")
    opener.add_argument("--agent", default=saved.CODER, choices=saved.AGENTS, help=argparse.SUPPRESS)
    opener.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")

    add_agent_arguments(parser)
    return parser


#: The subcommand flags that take a separate value, which `config_argv` has to
#: keep attached to their flag when it reorders the line. One set for every
#: subcommand it reorders: a flag missing from here would have its value hoisted
#: away from it and read as a positional, which is a worse bug than the one the
#: reordering exists to fix.
CONFIG_VALUED_FLAGS = {"--cwd", "--agent"}


def add_config_arguments(parser) -> None:
    """Flags for `coder config`.

    The verb and its arguments are positionals rather than sub-subcommands: the
    grammar is small enough that argparse's own would only get in the way, and
    a written help page says what it is anyway.
    """
    parser.add_argument("action", nargs="?", default="list", help="list, get, set, unset, or path.")
    parser.add_argument("rest", nargs="*", help="The key, and the value for `set`.")
    parser.add_argument("--workspace", action="store_true", help="Write this workspace's config, not the personal one.")
    parser.add_argument("--json", action="store_true", help="Emit the settings as JSON.")
    parser.add_argument("--cwd", default=".", help="Workspace root. Defaults to the current directory.")


def saved_cli_arguments(parser) -> None:
    """Flags for `coder sessions`, which needs a workspace and nothing else."""
    parser.add_argument("--all", action="store_true", help="Every workspace, not just this one.")
    parser.add_argument("--json", action="store_true", help="Emit the listing as JSON.")
    parser.add_argument(
        "--agent", default=saved.CODER, choices=saved.AGENTS,
        help="Whose sessions to list. Other agents' are read-only.",
    )
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
    """Which permission mode this run starts in.

    Both switches are unset when neither was typed *and* nothing was configured,
    which is what `ask` means. ``config.Settings.apply`` has already turned a
    configured mode into whichever of the two says it.
    """
    if args.auto and args.plan:
        raise SystemExit("--auto and --plan contradict each other; pick one.")
    if args.auto:
        return AUTO
    if args.plan:
        return PLAN
    return ASK


def configured(args):
    """Fill in whatever the user did not type, from the config files.

    Called on every path that parses arguments, so that `doctor` and `review`
    keep reading concrete values out of ``args`` and never learn that settings
    can come from anywhere else.

    A file that could not be read is announced rather than swallowed, for the
    same reason an unreadable permissions file is: a session that believes it
    is running the model you configured, and is not, is the one case where the
    next surprise is unattributable.
    """
    settings = config.load(Path(args.cwd).expanduser().resolve())
    if settings.error:
        print(paint(settings.error, YELLOW, sys.stderr), file=sys.stderr)
    settings.apply(args)
    return args


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
    # Here rather than at the call site, so that every caller -- `main`, the
    # tests, anything embedding this later -- gets the configured values. It
    # only fills flags nobody typed, so calling it twice changes nothing.
    configured(args)
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
    # spinner repaints over the diff being approved. It pauses the Esc watch
    # for the same stretch, or the keypress answering the question would be
    # read by the thread waiting to cancel the turn instead.
    watch = getattr(listener, "watch", None)
    pause = _pauses(
        listener.status.pause if getattr(listener, "status", None) else None,
        watch.pause if watch is not None else None,
    )
    permissions = Permissions(
        mode=mode, workspace=workspace,
        **({"prompt": partial(ask_tty, pause=pause)} if terminal.interactive() else {}),
    )
    # A permissions file that could not be read is announced, never swallowed:
    # a session that believes it has rules and does not is the one case where
    # the user would be surprised by a prompt they thought they had answered.
    if permissions.rules is not None and permissions.rules.error:
        print(paint(permissions.rules.error, YELLOW, sys.stderr), file=sys.stderr)
    provider = build_provider(
        getattr(args, "provider", DEFAULT_PROVIDER),
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


def _pauses(*managers):
    """One context manager entering each of the given ones, in order."""
    managers = [manager for manager in managers if manager is not None]
    if not managers:
        return None

    @contextlib.contextmanager
    def paused():
        with contextlib.ExitStack() as stack:
            for manager in managers:
                stack.enter_context(manager())
            yield

    return paused


def run_turn(agent, listener, task: str, images: list[str] | None = None) -> int:
    """One task, with the spinner and the gate bounded to it."""
    if listener is None:
        return report(agent.run(task, images=images))
    with listener.turn():
        outcome = agent.run(task, images=images)
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


class StatusLine:
    """The fields of the row under the prompt, gathered on demand.

    A class rather than a lambda because one of the fields costs a subprocess.
    The row is rebuilt on every keystroke, and asking git which branch this is a
    hundred times a line would make typing wait on a fork -- so the branch is
    read once and again whenever a line is submitted, which is the only moment
    in a session where it plausibly changed.
    """

    def __init__(self, agent, workspace, note: str = "") -> None:
        self.agent = agent
        self.workspace = workspace
        self.note = note
        self.branch = git.branch(workspace.root)

    def refresh(self) -> None:
        self.branch = git.branch(self.workspace.root)

    def fields(self) -> dict:
        session = self.agent.session
        num_ctx = getattr(self.agent.provider, "num_ctx", DEFAULT_NUM_CTX)
        return {
            "name": self.workspace.root.name or str(self.workspace.root),
            "branch": self.branch,
            "ratio": usage_ratio(session, num_ctx),
            "model": self.agent.provider.model,
            "spent": session.prompt_tokens + session.completion_tokens,
            "width": terminal.width(),
        }


def pinned_block(permissions, line, stream=None):
    """The prompt block, as rows the spinner can keep on screen through a turn.

    The same five rows the editor draws, with the input empty: the line that
    was typed has already scrolled into the transcript above, and what stays is
    the frame it was typed into. Before this, submitting a line took the whole
    block off the screen and a turn ran against a bare spinner -- the session
    lost its shape at exactly the moment there was most to say about it.

    Empty rather than echoing the question: this is not an editor and nothing
    typed into it would be read, so a box with words in it would be a promise
    the turn cannot keep.

    Painted and fitted here, because :class:`~bkht.coder.status.Status` stacks
    rows and counts them and must not have to know what is in one.
    """

    def rows() -> list[str]:
        stream_ = sys.stdout if stream is None else stream
        width = terminal.width()
        rule = paint(banner.rule(width), DIM, stream_)
        footer = lineedit.footer_rows(permissions.mode, **line.fields())
        above = lineedit.aside(line.note, width, stream_)
        return [
            *([above] if above else []),
            rule,
            paint(PROMPT.rstrip(), ACCENT + BOLD, stream_),
            rule,
            *footer.split("\n"),
        ]

    return rows


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


def greeting(agent, permissions, workspace, stream=None, loaded=None, notice="") -> str:
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
    plain = "\n".join(filter(None, (
        header(agent, permissions, workspace), notice, HINT,
    )))
    if not terminal.interactive(stream):
        return plain
    if terminal.width() < banner.MIN_WIDTH or not banner.drawable(stream):
        return paint(plain, DIM, stream)

    # The mode and the context count used to sit here too. They are the two
    # facts on screen that change while the session runs, and a greeting is
    # scrollback -- it said `ask` for the rest of the session no matter how
    # many times Shift+Tab was pressed. Both are now on the row under the
    # prompt, which is redrawn on the keypress that changes them.
    #
    # Only the name is coloured. What a greeting is for is being read once and
    # then ignored, and a column of accents is a column that keeps asking.
    return banner.render([
        paint(" ".join(filter(None, ("bkht.coder", version()))), ACCENT, stream),
        paint(agent.provider.model, DIM, stream),
        paint(home_relative(workspace.root), DIM, stream),
        *(paint(line, DIM, stream) for line in (loaded.lines() if loaded else [])),
        # Last, and dim like the rest: a release is worth mentioning once, not
        # worth being the brightest thing in the box.
        *([paint(notice, DIM, stream)] if notice else []),
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


def attach_image(stream=None) -> str | None:
    """Take an image off the clipboard and save it. None when there is none.

    The complaint, not the picture, is what this has to get right: a keypress
    that silently does nothing is worse than no keypress at all, so a clipboard
    with no image in it and a box with no helper installed say different things.
    """
    stream = sys.stdout if stream is None else stream
    if missing := clipboard.helper_missing():
        print(paint(f"  ! image paste needs {missing}", YELLOW, stream), file=stream)
        return None
    data = clipboard.read_image()
    if not data or not clipboard.looks_like_png(data):
        print(paint("  ! no image on the clipboard", DIM, stream), file=stream)
        return None
    try:
        return str(clipboard.save(data))
    except OSError as exc:
        print(paint(f"  ! could not save the image: {exc}", YELLOW, stream), file=stream)
        return None


def announce_image(provider, path: str, stream=None) -> None:
    """Say whether the model is going to look at what was just attached.

    Said at the moment of pasting rather than after the answer comes back. A
    model that cannot see silently ignores the picture and answers from the
    text, which reads as an answer about the image -- and the user has no way
    to know it was never looked at.
    """
    stream = sys.stdout if stream is None else stream
    if getattr(provider, "can_see", lambda: False)():
        print(paint(f"  attached {path}", DIM, stream), file=stream)
        return
    print(paint(f"  ! {provider.model} cannot see images.", YELLOW, stream), file=stream)
    print(paint(f"    Saved to {path} -- the path is in the prompt.", DIM, stream), file=stream)
    print(paint("    /model qwen2.5vl:7b for one that can.", DIM, stream), file=stream)


def interactive(agent, snapshots, permissions, workspace, listener,
                use_instructions=True, jobs=None, loaded=None, notice="") -> int:
    """The REPL. Ctrl-C abandons the current line; Ctrl-D leaves."""
    repl = Repl(
        agent, snapshots, permissions, workspace,
        use_instructions=use_instructions, jobs=jobs,
    )
    # The footer is a callable, not a string: the mode it names is the thing
    # Shift+Tab changes, and the editor redraws it on the same keypress -- and
    # the row above it counts tokens that change with every turn.
    line = StatusLine(agent, workspace, note=notice)
    reader = Reader(
        repl,
        enabled=terminal.interactive(),
        footer=lambda: lineedit.footer_rows(permissions.mode, **line.fields()),
        header=lambda: lineedit.aside(line.note, terminal.width()),
        cycle=lambda: setattr(permissions, "mode", next_mode(permissions.mode)),
        attach=attach_image,
        on_image=lambda path: announce_image(agent.provider, path),
    )
    # Esc stops a running turn. Held here rather than inside `run_turn`,
    # because the approval prompt has to be able to borrow the terminal from
    # it, and only the caller holding both can arrange that.
    watch = cancel.Watch(enabled=cancel.available())
    listener.watch = watch
    if getattr(listener, "status", None) is not None:
        listener.status.cancellable = watch.enabled
        # The block the spinner keeps on screen for the length of a turn. Only
        # here: a one-shot run prints an answer and leaves, and a prompt frame
        # around nothing is chrome for a session that is not going to happen.
        listener.status.block = pinned_block(permissions, line)

    # Printed as it comes back: the greeting dims and bolds its own parts now,
    # and a paint() around the whole of it would flatten both.
    print(greeting(agent, permissions, workspace, loaded=loaded, notice=notice))

    while True:
        try:
            # The editor opens with a rule of its own; a divider above it
            # would be the same line drawn twice.
            if rule := (divider() if not reader.cycles else ""):
                print(rule)
                # Where there is no editor -- Windows, a pipe, an IDE console
                # -- the same two rows are printed above the input instead of
                # under it. Above is the only side readline leaves free, and
                # the rows are worth more out of place than absent.
                if rows := lineedit.footer_rows(
                    permissions.mode, cycles=False, **line.fields()
                ):
                    print(rows)
            elif reader.cycles:
                print()
            text = read_line(reader, PROMPT)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        # The branch can have changed while the last turn ran -- a checkout in
        # another window, or one this session made itself.
        line.refresh()
        command = repl.dispatch(text)
        if command.quit:
            return 0
        if command.handled:
            continue

        try:
            with watch.watching():
                run_turn(agent, listener, command.task, images=reader.images())
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


def config_argv(argv: list[str]) -> list[str]:
    """A subcommand line with its positionals moved ahead of its flags.

    argparse before CPython 3.12.7 cannot fill an optional positional that
    appears *after* a flag, and `config` is the one subcommand whose grammar is
    positionals -- `set <key> <value>` -- rather than options. So
    `config set --workspace num_ctx 8192` parsed here and died on a distro
    python with "unrecognized arguments: num_ctx 8192".

    Reordering rather than raising `requires-python`: the shape argparse has
    always handled is positionals first, and every interpreter this package
    claims to support can parse that.
    """
    head: list[str] = []
    flags: list[str] = []
    index = 1  # argv[0] is the subcommand name, which stays put
    while index < len(argv):
        token = argv[index]
        if token.startswith("-"):
            flags.append(token)
            # The only `config` flag that takes a separate value. Written as a
            # set so a second one cannot be added to the parser and forgotten
            # here without the name being obvious.
            if token in CONFIG_VALUED_FLAGS and index + 1 < len(argv):
                index += 1
                flags.append(argv[index])
        else:
            head.append(token)
        index += 1
    return [argv[0], *head, *flags]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv[:1] == ["review"]:
        return review_cli.run(configured(build_parser().parse_args(argv)))

    if argv[:1] == ["config"]:
        return config.run(build_parser().parse_args(config_argv(argv)))

    if argv[:1] == ["help"] or argv[:1] == ["--help"] or argv[:1] == ["-h"]:
        print(usage.HELP)
        return 0

    if argv[:1] == ["sessions"]:
        args = build_parser().parse_args(config_argv(argv))
        return saved.report(
            Path(args.cwd).expanduser().resolve(),
            as_json=args.json, everywhere=args.all, agent=args.agent,
        )

    if argv[:1] == ["session"]:
        # `resume` re-enters the ordinary agent path rather than reimplementing
        # it, so a resumed session gets the current tools, instructions and
        # permission mode like any other.
        rest = argv[1:]
        if rest[:1] == ["resume"]:
            argv = resume_argv(rest[1:])
        else:
            args = build_parser().parse_args(config_argv(argv))
            # `claude/1be46299` names somebody else's transcript, which is read
            # rather than resumed; anything else is one of ours.
            if "/" in (args.target or ""):
                return saved.show_agent(args.target, as_json=args.json)
            return saved.show(
                Path(args.cwd).expanduser().resolve(),
                args.target, as_json=args.json,
            )

    if argv[:1] == ["update"]:
        return update.run(build_parser().parse_args(argv))

    if argv[:1] == ["doctor"]:
        args = configured(build_parser().parse_args(argv))
        return doctor.report(
            Path(args.cwd).expanduser().resolve(),
            model=args.model, host=args.host, num_ctx=args.num_ctx,
            provider=args.provider, as_json=args.json,
        )

    args = build_agent_parser().parse_args(argv)
    listener = TerminalListener(verbose=args.verbose)
    agent, snapshots, permissions, workspace, jobs, announced = make_agent(args, listener)
    # Started here rather than in `make_agent`, which the tests call: nothing
    # should spawn a thread that talks to the network just by building an
    # agent. The notice it fills is for the next session, not this one.
    settings = config.load(Path(args.cwd).expanduser().resolve())
    update.start(settings)

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
            notice=update.notice(settings),
        )
    finally:
        # The user started nothing here and can see nothing here: a server left
        # running after the session ends is one they have no way to find again.
        jobs.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
