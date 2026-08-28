"""Interactive session: slash commands, shell escapes, and the prompt loop.

Kept out of ``cli.py`` so argument parsing and wiring stay readable, and so the
command table can be tested by calling :meth:`Repl.dispatch` directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from . import config, highlight
from .commands import discover as user_commands, summarize as summarize_commands
from .instructions import load_instructions, render
from .permissions import MODES
from .provider import DEFAULT_NUM_CTX
from .rules import DECISIONS
from .skills import discover as discover_skills
from .tools.base import ToolError, set_output_budget, validate_arguments
from .tools.shell import NO_SHELL, resolve_shell

HELP = """\
Commands
  /tools              list the tools available to the model
  /context            token usage, message count, and session file
  /clear              forget the conversation, keep the workspace
  /undo               restore the file changed by the last mutating call
  /diff               show uncommitted changes in the workspace
  /instructions [reload]  show the project instructions, or re-read them
  /skills             list the skills the model can open
  /jobs [stop <id>]   background processes this session started
  /review [base]      review uncommitted changes, or this branch against base
  /model [name]       show or switch the Ollama model
  /mode [ask|auto|plan]   show or switch the permission mode (or shift+tab)
  /config [set|unset] show or change the settings that outlive this session
  /permissions        list remembered decisions, or `remember`/`revoke` one
  /doctor             check that this install can run a turn
  /help               this list
  /exit               leave (or just `exit`)

  !<command>          run a shell command yourself, without the model"""


@dataclass
class Command:
    """The result of handling one line of input."""

    handled: bool = True
    quit: bool = False
    task: str | None = None  # a line that should go to the model instead


class Repl:
    """Handles everything typed at the prompt that is not a task."""

    def __init__(
        self, agent, snapshots, permissions, workspace, out=print,
        use_instructions: bool = True, jobs=None,
    ) -> None:
        self.agent = agent
        self.snapshots = snapshots
        self.permissions = permissions
        self.workspace = workspace
        self.out = out
        self.use_instructions = use_instructions
        self.jobs = jobs

    LEAVE = ("exit", "quit")

    def dispatch(self, line: str) -> Command:
        """Route one input line: slash command, shell escape, or a task."""
        line = line.strip()
        if not line:
            return Command()

        # Typed bare, without the slash, because that is what leaves every
        # other prompt -- and sending it to the model instead is a slow,
        # confusing way to not quit.
        if line.lower() in self.LEAVE:
            return Command(quit=True)

        if line.startswith("!"):
            self.shell(line[1:].strip())
            return Command()

        if not line.startswith("/"):
            return Command(handled=False, task=line)

        name, _, argument = line[1:].partition(" ")
        handler = getattr(self, f"do_{name}", None)
        if handler is not None:
            return handler(argument.strip()) or Command()

        # Looked up only after the built-ins, so a file on disk can never
        # shadow /undo or /mode -- a command that quietly stops meaning what it
        # has always meant is worse than one that does not exist.
        written = user_commands(self.workspace.root).get(name.lower())
        if written is not None:
            return Command(handled=False, task=written.expand(argument))

        self.out(f"Unknown command /{name}. Try /help.")
        return Command()

    # --- commands -----------------------------------------------------------

    def do_help(self, argument: str) -> Command:
        self.out(HELP + summarize_commands(user_commands(self.workspace.root)))
        return Command()

    def do_exit(self, argument: str) -> Command:
        return Command(quit=True)

    do_quit = do_exit

    def do_tools(self, argument: str) -> Command:
        for tool in self.agent.registry:
            marker = " (needs permission)" if tool.mutating else ""
            self.out(f"  {tool.name}{marker}\n      {tool.description}")
        return Command()

    def do_instructions(self, argument: str) -> Command:
        """Show the project instructions, or re-read them from disk.

        Reload exists because editing CLAUDE.md mid-session is the common case,
        and without it the only way to apply the edit is to restart and lose
        the conversation.
        """
        if not self.use_instructions:
            self.out("Instructions are disabled for this session (--no-instructions).")
            return Command()

        loaded = load_instructions(self.workspace.root)
        if not loaded:
            self.out("No AGENTS.md or CLAUDE.md found for this workspace.")
            return Command()

        if argument.strip() == "reload":
            from .context import file_tree
            from .prompts import system_prompt

            # Rebuilt whole, so the skill listing has to be rebuilt with it --
            # reloading instructions must not quietly cost the session its
            # skills.
            from .skills import render as render_skills

            self.agent.session.system = system_prompt(
                self.agent.registry,
                str(self.workspace.root),
                file_tree(self.workspace.root),
                render(loaded),
                render_skills(discover_skills(self.workspace.root)),
            )
            self.out(f"Reloaded {len(loaded)} instruction file(s).")

        for instruction in loaded:
            marker = " (truncated)" if instruction.truncated else ""
            self.out(f"  {instruction.source}{marker}")
            for line in instruction.text.splitlines():
                self.out(f"      {line}")
        return Command()

    def do_skills(self, argument: str) -> Command:
        """Show which skills were found, and which were skipped and why."""
        found = discover_skills(self.workspace.root)
        if not (found.skills or found.problems):
            self.out(
                "No skills found. Add one as "
                ".bkht-coder/skills/<name>/SKILL.md with a name and description "
                "in its frontmatter."
            )
            return Command()

        for skill in found.skills:
            available = " (loaded)" if "skill" in self.agent.registry else ""
            self.out(f"  {skill.name}{available}\n      {skill.description}\n      {skill.source}")
        for problem in found.problems:
            self.out(f"  skipped: {problem}")
        return Command()

    def do_doctor(self, argument: str) -> Command:
        """The same checks as `coder doctor`, against this session's settings."""
        from . import doctor

        provider = self.agent.provider
        doctor.report(
            self.workspace.root,
            model=provider.model,
            host=getattr(provider, "host", ""),
            num_ctx=getattr(provider, "num_ctx", DEFAULT_NUM_CTX),
            out=self.out,
        )
        return Command()

    def do_jobs(self, argument: str) -> Command:
        """List background processes, or stop one.

        Visible to the user, not just to the model: a process the agent started
        and the user cannot see is one they cannot decide about.
        """
        if self.jobs is None:
            self.out("Background jobs are not available in this session.")
            return Command()

        verb, _, identifier = argument.partition(" ")
        if not verb:
            self.out(self.jobs.summary())
            return Command()
        if verb != "stop" or not identifier.strip():
            self.out("Usage: /jobs\n       /jobs stop <id>")
            return Command()

        try:
            self.out(self.jobs.stop(identifier.strip()))
        except ToolError as exc:
            self.out(str(exc))
        return Command()

    def do_context(self, argument: str) -> Command:
        session = self.agent.session
        num_ctx = getattr(self.agent.provider, "num_ctx", DEFAULT_NUM_CTX)
        used = session.prompt_tokens
        percent = f"{100 * used / num_ctx:.0f}%" if num_ctx else "?"

        self.out(f"  model      {self.agent.provider.model}")
        self.out(f"  mode       {self.permissions.mode}")
        self.out(f"  messages   {len(session.messages)}")
        self.out(f"  context    {used} / {num_ctx} prompt tokens ({percent})")
        self.out(f"  generated  {session.completion_tokens} tokens")
        self.out(f"  session    {session.path or 'not saved'}")
        self.out(f"  undo depth {len(self.snapshots)}")
        self.out(f"  scout      {'on' if self.agent.scout_root else 'off'}")
        self.out(f"  skills     {len(discover_skills(self.workspace.root))} available")
        if self.jobs is not None:
            self.out(f"  jobs       {len(self.jobs.running())} running")
        rules = getattr(self.permissions, "rules", None)
        if rules is not None:
            self.out(f"  remembered {len(rules.listing())} permission rule(s)")
        return Command()

    def do_clear(self, argument: str) -> Command:
        self.agent.session.clear()
        self.out("Conversation cleared. Files are untouched.")
        return Command()

    def do_undo(self, argument: str) -> Command:
        what = self.snapshots.undo()
        self.out(what if what else "Nothing to undo.")
        return Command()

    def do_diff(self, argument: str) -> Command:
        try:
            completed = subprocess.run(
                ["git", "diff"],
                cwd=self.workspace.root,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.out(f"Could not run git diff: {exc}")
            return Command()

        if completed.returncode != 0:
            self.out((completed.stderr or "git diff failed").strip())
        elif completed.stdout.strip():
            # The same renderer the approval preview uses, so a change looks
            # the same whether it is being approved or reviewed after the fact.
            self.out(highlight.diff(completed.stdout.rstrip(), colour=sys.stdout.isatty()))
        else:
            self.out("No uncommitted changes.")
        return Command()

    def do_review(self, argument: str) -> Command:
        """Review the working tree, or this branch against a base branch."""
        from .review import render
        from .review.cli import Progress
        from .review.diff import GitError, collect_diff
        from .review.reviewer import Reviewer

        try:
            files = collect_diff(self.workspace.root, base=argument or None)
        except GitError as exc:
            self.out(str(exc))
            return Command()

        if not files:
            self.out("Nothing to review: no changes were found.")
            return Command()

        from .provider import for_review

        reviewer = Reviewer(
            for_review(self.agent.provider), self.workspace.root, listener=Progress()
        )
        self.out(render.terminal(reviewer.review(files), colour=False))
        return Command()

    def do_model(self, argument: str) -> Command:
        if not argument:
            self.out(self.agent.provider.model)
            return Command()
        self.agent.provider.model = argument
        self.agent.session.model = argument
        self.out(f"Model is now {argument}.")
        return Command()

    def do_mode(self, argument: str) -> Command:
        if not argument:
            self.out(self.permissions.mode)
            return Command()
        if argument not in MODES:
            self.out(f"Unknown mode {argument!r}. Expected one of {', '.join(MODES)}.")
            return Command()

        self.permissions.mode = argument
        self.out(f"Permission mode is now {argument}.")
        return Command()

    CONFIG_USAGE = (
        "Usage: /config\n"
        "       /config set <key> <value> [--workspace]\n"
        "       /config unset <key> [--workspace]"
    )

    def do_config(self, argument: str) -> Command:
        """Show the settings in force, or change one for good.

        `/model` and `/mode` change this session and forget it at the prompt;
        this writes the change down. Which is also why it says, every time,
        whether the running session took the new value or whether it starts
        applying at the next one -- a setting that was stored but is not in
        force is exactly the kind of thing that gets debugged twice.
        """
        words = argument.split()
        scope = config.WORKSPACE if "--workspace" in words else config.GLOBAL
        words = [word for word in words if word != "--workspace"]
        root = self.workspace.root

        if not words:
            settings = config.load(root)
            if settings.error:
                self.out(settings.error)
            self.out(config.render(settings))
            return Command()

        verb, rest = words[0], words[1:]
        try:
            if verb == "set":
                if len(rest) < 2:
                    raise config.ConfigError(f"set takes a key and a value.\n{self.CONFIG_USAGE}")
                # Joined, so a value with a space in it is not silently halved.
                key = rest[0]
                config.set_value(key, " ".join(rest[1:]), scope=scope, root=root)
            elif verb == "unset":
                if len(rest) != 1:
                    raise config.ConfigError(f"unset takes one key.\n{self.CONFIG_USAGE}")
                key = rest[0]
                if not config.unset(key, scope=scope, root=root):
                    self.out(f"{key} was not set in the {scope} config.")
                    return Command()
            else:
                self.out(self.CONFIG_USAGE)
                return Command()
        except config.ConfigError as exc:
            self.out(str(exc))
            return Command()

        # Read back rather than assumed: unsetting a workspace value leaves
        # whatever the global file or the default says, and that is the value
        # the session should now be running.
        value = config.load(root).values[key]
        self.out(
            f"{key} is now {config.format_value(value)} ({scope} config). "
            + self._adopt(key, value)
        )
        return Command()

    def _adopt(self, key: str, value) -> str:
        """Apply a setting to the running session, and say whether it took.

        Three of them cannot be applied: the provider is constructed once, and
        instructions and skills are baked into the system prompt before the
        first turn. Saying so is better than a silent no-op -- and better than
        rebuilding half a session under the user, which is what a restart is
        for.
        """
        if not config.BY_NAME[key].live:
            return "Takes effect in the next session."

        provider = self.agent.provider
        if key == "model":
            provider.model = value
            self.agent.session.model = value
        elif key == "host":
            provider.host = str(value).rstrip("/")
        elif key == "num_ctx":
            provider.num_ctx = value
            # The tool-output cap is a share of the window, so it moves with it.
            set_output_budget(value)
        elif key == "temperature":
            provider.temperature = value
        elif key == "mode":
            self.permissions.mode = value
        elif key == "scout":
            self.agent.scout_root = self.workspace.root if value else None
        elif key == "max_iterations":
            self.agent.max_iterations = value
        return "Applied to this session."

    PERMISSIONS_USAGE = (
        "Usage: /permissions\n"
        "       /permissions remember allow|deny <tool> <arguments-json>\n"
        "       /permissions revoke <rule-id>"
    )

    def do_permissions(self, argument: str) -> Command:
        """Show, add, or remove remembered permission decisions.

        `remember` exists so a decision can be made deliberately, at a keyboard,
        rather than only in the middle of a turn -- which is when the user is
        least inclined to read what they are approving.
        """
        rules = getattr(self.permissions, "rules", None)
        if rules is None:
            self.out("No permission store for this session.")
            return Command()

        verb, _, rest = argument.partition(" ")
        rest = rest.strip()

        if not verb:
            return self._list_rules(rules)
        if verb == "revoke":
            return self._revoke_rule(rules, rest)
        if verb == "remember":
            return self._remember_rule(rules, rest)

        self.out(self.PERMISSIONS_USAGE)
        return Command()

    def _list_rules(self, rules) -> Command:
        stored = rules.listing()
        if not stored:
            self.out("No remembered decisions for this workspace.")
            return Command()
        for rule in stored:
            self.out("  " + rule.label())
        return Command()

    def _revoke_rule(self, rules, identifier: str) -> Command:
        if not identifier:
            self.out(self.PERMISSIONS_USAGE)
            return Command()
        removed = rules.revoke(identifier)
        self.out(
            f"Revoked {removed.tool} rule {identifier}."
            if removed
            else f"No rule with id {identifier}. /permissions lists them."
        )
        return Command()

    def _remember_rule(self, rules, rest: str) -> Command:
        decision, _, tail = rest.partition(" ")
        name, _, payload = tail.strip().partition(" ")
        if decision not in DECISIONS or not name or not payload.strip():
            self.out(self.PERMISSIONS_USAGE)
            return Command()

        tool = self.agent.registry.get(name)
        if tool is None:
            self.out(f"No tool named {name}. /tools lists them.")
            return Command()

        try:
            arguments = json.loads(payload)
        except ValueError as exc:
            self.out(f"Arguments must be JSON: {exc}")
            return Command()

        # Validated against the tool's own schema: a rule whose arguments the
        # tool would reject can never match a real call, and a stored rule that
        # silently never fires is worse than no rule at all.
        try:
            arguments = validate_arguments(tool, arguments)
        except ToolError as exc:
            self.out(str(exc))
            return Command()

        rule = rules.remember(tool.name, arguments, decision)
        self.out(f"Remembered: {rule.label()}")
        return Command()

    # --- shell escape -------------------------------------------------------

    def shell(self, command: str) -> None:
        """Run a command as the user, not the model. Not permission-gated."""
        if not command:
            self.out("Usage: !<command>")
            return
        # `shell=True` would run through COMSPEC -- cmd.exe -- on Windows, so
        # `!cmd` and the model's shell tool would disagree about the syntax.
        shell = resolve_shell()
        if shell is None:
            self.out(f"Could not run command: {NO_SHELL}")
            return
        try:
            subprocess.run([*shell.argv, command], cwd=self.workspace.root)
        except OSError as exc:
            self.out(f"Could not run command: {exc}")
