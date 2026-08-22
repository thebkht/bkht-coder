"""Help pages, written rather than generated.

argparse can describe this program, but only as one flat list: the subcommands
arrive as a `{review,doctor}` choice line, the flags arrive in declaration
order, and nothing says that `coder` on its own starts a session. That is a
reference; what someone typing `--help` wants is an introduction.

So the pages are literals, the way ``repl.HELP`` is, and the parsers serve them
through ``format_help`` -- which means ``-h``, ``print_help()`` and argparse's
own "try --help" error path all show the same written page, with no second
rendering to keep in step.

The cost is that a flag added below must be added to the text too. That is the
point: a flag nobody wrote a line for is a flag nobody has explained.
"""

from __future__ import annotations

import argparse

from .doctor import version

TAGLINE = "A coding agent, on your own machine."


def title() -> str:
    """``coder v0.1.0``, or plain ``coder`` from a source checkout."""
    release = version()
    return f"coder v{release}" if release else "coder"


#: Flags every entry point takes, so they are described in one place only.
COMMON = """\
  --model <name>          Ollama model to use
  --host <url>            Ollama server URL
  --num-ctx <n>           Context window to request
  --cwd <path>            Workspace root; defaults to the current directory
  --auto                  Allow every tool call without prompting
  --no-instructions       Ignore AGENTS.md and CLAUDE.md
  --no-skills             Ignore skills, and omit the skill tool"""

FOOTER = """\
Run `coder <command> --help` for command-specific options.
Run `/help` inside an interactive session for slash commands."""

HELP = f"""\
{title()}
{TAGLINE}

coder starts an interactive session by default. Give it a prompt to run one
task and exit instead.

Usage:
  coder [flags] [prompt]
  coder <command> [...flags] [...args]

Commands:
  <prompt>                       Run one task and exit

  sessions                       List saved sessions for this workspace
  session <last|id>              Show a saved session
  session resume [last|id]       Continue a saved session

  review [range]                 Review changes, a branch, or a commit range
  doctor                         Check that this install can run a turn
  help                           Show this help

Flags:
  --resume [last|<id>]    Continue a saved session; the newest by default
  --plan                  Read-only: refuse every change to the workspace
  --verbose               Stream raw model output and tool results
  --no-scout              Do not search the workspace before each task
  --max-iterations <n>    Cap on loop iterations per task
{COMMON}
  -h, --help              Display this help and exit
  -v, --version           Print the coder version and exit

Examples:
  coder
      Start an interactive session in the current directory

  coder "explain the changes in this repository"
      Run one task and exit

  coder session resume last
      Continue the latest session for this workspace

  coder review --base main
      Review this branch against its merge base with main

{FOOTER}"""

SESSIONS_HELP = f"""\
coder sessions
List saved sessions, newest first.

Sessions are append-only files under ~/.bkht-coder/sessions, written as each
turn happens, so a session killed mid-answer is still listed and still
resumable up to its last complete message.

Usage:
  coder sessions [flags]

Flags:
  --all                   Every workspace, not just this one
  --json                  Emit the listing as JSON
  --cwd <path>            Workspace root; defaults to the current directory
  -h, --help              Display this help and exit

Examples:
  coder sessions
      What is saved for the current directory

  coder sessions --all
      Every session on this machine, with the directory it belongs to
"""

SESSION_HELP = f"""\
coder session
Show or resume one saved session.

`last` means the most recent session for this workspace. An id can be given in
full or as a prefix long enough to be unambiguous.

Usage:
  coder session <last|id> [flags]
  coder session resume [last|id] [flags]

Flags:
  --json                  Emit the session as JSON
{COMMON}
  -h, --help              Display this help and exit

Examples:
  coder session last
      The tail of the latest transcript, without resuming it

  coder session resume 20260822-101455
      Continue that session, with the current tools and instructions

  coder sessions
      List what there is to resume
"""

REVIEW_HELP = f"""\
coder review
Review uncommitted changes, a branch, or a commit range.

The review itself is read-only. --fix is a second phase you opt into, and it
goes through the normal permission gate, so nothing is written unapproved.

Usage:
  coder review [range] [flags]

Arguments:
  range                   Commit range, e.g. HEAD~3..HEAD

Flags:
  --base <ref>            Review this branch from its merge base with <ref>
  --staged                Review staged changes only
  --files <path>...       Review whole files rather than a diff
  --dimension <name>      Limit to one dimension; repeatable
  --no-verify             Skip the refutation pass; expect false positives
  --json                  Emit the findings as JSON
  --output <path>         Write a Markdown report to this path
  --fix                   After reporting, offer to fix findings
  --quiet                 Suppress progress output
{COMMON}
  -h, --help              Display this help and exit

Examples:
  coder review
      Review what is uncommitted in the workspace

  coder review --base main
      Review this branch against its merge base with main

  coder review HEAD~3..HEAD --json
      Review three commits and emit the findings as JSON
"""

DOCTOR_HELP = f"""\
coder doctor
Check that this install can actually run a turn.

Checks the things that make a session fail on its first answer rather than at
startup: the server, the model, the context window, and the workspace.

Usage:
  coder doctor [flags]

Flags:
  --json                  Machine-readable output for CI
{COMMON}
  -h, --help              Display this help and exit

Examples:
  coder doctor
      Check this install before trusting it with a task

  coder doctor --json
      The same checks, for CI to assert on
"""


class Parser(argparse.ArgumentParser):
    """An argparse parser whose help page is written, not generated."""

    def __init__(self, *args, page: str = "", **kwargs) -> None:
        self.page = page
        super().__init__(*args, **kwargs)

    def format_help(self) -> str:
        return self.page if self.page else super().format_help()
