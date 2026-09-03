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
#: Each of them overrides the matching key in the config files; `coder config`
#: shows what those are set to and where each value came from.
COMMON = """\
  --provider <name>       Backend: local, ollama, claude-code, codex
  --model <name>          Model to use
  --host <url>            Model server URL (local and ollama)
  --num-ctx <n>           Context window to request
  --temperature <n>       Sampling temperature; low keeps tool calls valid
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

  config [get|set|unset]         Show or change settings that survive a restart
  update [--check]               Install the newest release, if there is one

  review [range]                 Review changes, a branch, or a commit range
  doctor                         Check that this install can run a turn
  dataset [build|stats|show]     Build a training corpus from transcripts
  help                           Show this help

Flags:
  --resume [last|<id>]    Continue a saved session; the newest by default
  --plan                  Read-only: refuse every change to the workspace
  --verbose               Stream raw model output and tool results
  --no-scout              Do not search the workspace before each task
  --no-planning           Omit the plan tool
  --no-delegation         Omit the task tool, which runs a sub-agent
  --verify-command <cmd>  Test command to run after a turn's edits
  --no-verify             Do not run it for this session
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

CONFIG_HELP = """\
coder config
Show or change the settings that survive a restart.

Settings resolve lowest to highest: the built-in default, then your personal
file, then this workspace's, then any flag you typed -- so a flag always wins
for the run you are in, and a workspace can pin what a project needs.

  ~/.bkht-coder/config.json      personal defaults, everywhere
  .bkht-coder/config.json        this workspace only, written with --workspace

`provider` names the model backend:

  local         any OpenAI-compatible server -- mlx_lm.server, llama-server,
                vLLM, or Ollama's own /v1. The default.
  ollama        Ollama's native API, for a stock model it has already pulled
  claude-code   the `claude` command, using the login it already has
  codex         the `codex` command, using the login it already has

The first two both keep the work on hardware you own. `local` is the default
because it is the one that can serve a model you trained yourself -- and
because `host` may name another machine on your network, so the box with the
memory can do the thinking while you drive from a laptop.

The last two are borrowed transports: their own file tools are switched off,
and every change still goes through coder's permission gate. Switching provider
brings that backend's own model and window with it unless you have pinned your
own.

Usage:
  coder config [list] [flags]
  coder config get <key>
  coder config set <key> <value> [flags]
  coder config unset <key> [flags]
  coder config path

Settings:
  provider                Model backend: local, ollama, claude-code, codex
  model                   Model to run
  host                    Model server URL (local and ollama)
  num_ctx                 Context window to plan for
  temperature             Sampling temperature
  mode                    Permission mode: ask, auto, plan
  scout                   Search the workspace before each task
  max_iterations          Cap on loop iterations per task
  instructions            Read AGENTS.md and CLAUDE.md
  skills                  Load skills, and offer the skill tool
  planning                Offer the plan tool
  delegation              Offer the task tool, which runs a sub-agent
  verify_command          Test command run after a turn's edits; off by default
  verify                  Run verify_command after a turn that changed files
  update_check            Ask once a day whether a newer release exists

Flags:
  --workspace             Write .bkht-coder/config.json, not the personal file
  --json                  Emit the settings as JSON
  --cwd <path>            Workspace root; defaults to the current directory
  -h, --help              Display this help and exit

Examples:
  coder config
      Every setting, its value, and which layer it came from

  coder config set model qwen2.5-coder:7b
      Use the smaller model everywhere from now on

  coder config set --workspace num_ctx 8192
      Pin a smaller window for this project only

  coder config unset model
      Fall back to whatever the next layer down says
"""

SESSIONS_HELP = """\
coder sessions
List saved sessions, newest first.

Sessions are append-only files under ~/.bkht-coder/sessions, written as each
turn happens, so a session killed mid-answer is still listed and still
resumable up to its last complete message.

Usage:
  coder sessions [flags]

Claude Code and Codex keep their own transcripts on this machine. `--agent`
reads those too, so what was already tried in a directory can be looked up
whatever was open at the time. Theirs are read-only: their agent holds state
this one has never seen, and a conversation replayed without it is not the
same session.

Flags:
  --all                   Every workspace, not just this one
  --agent <name>          coder (default), claude, codex, or all
  --json                  Emit the listing as JSON
  --cwd <path>            Workspace root; defaults to the current directory
  -h, --help              Display this help and exit

Examples:
  coder sessions
      What is saved for the current directory

  coder sessions --all
      Every session on this machine, with the directory it belongs to

  coder sessions --agent all
      Every agent's sessions for this directory, newest first

  coder session claude/1be46299
      Read one of them
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
  --ci [<name>]           CI output: log sections and inline annotations.
                          Detected from the environment; one of auto, github,
                          gitlab, generic, off
  --code-quality <path>   Write a GitLab Code Quality report to this path
{COMMON}
  -h, --help              Display this help and exit

Examples:
  coder review
      Review what is uncommitted in the workspace

  coder review --base main
      Review this branch against its merge base with main

  coder review HEAD~3..HEAD --json
      Review three commits and emit the findings as JSON

  coder review --base main --ci
      Annotate the diff on the pull request, and fail on a finding
"""

DATASET_HELP = """\
coder dataset
Build the training corpus from the transcripts already on this machine.

A coding agent produces exactly what a coding agent has to be trained on: a
task, the calls made to work it out, what each returned, and an answer. This
collects those from coder's own sessions and from Claude Code's and Codex's,
translates every call into coder's protocol, and writes the chat JSONL that
`mlx_lm.lora` reads.

Translation is not cosmetic. A model trained on `Read(file_path=...)` learns to
emit a call coder has no tool for; the same call mapped to `read_file(path=...)`
teaches the one that exists. Arguments coder's tools do not accept are dropped,
because an unknown argument is a hard error at runtime, not a warning. Calls
with no coder equivalent end the trajectory there rather than leaving a hole in
the middle of it.

Two things are done to match what the model will actually see at serve time:
tool results are cut to coder's own output budget, and an example too long for
the training window has its older results elided exactly as a long session's
are. Both keep the training bytes and the served bytes the same shape.

`build` writes the files, `stats` says what is in them, and `show` prints one
example as the model reads it. Run `show` at least once: a histogram cannot
tell you a conversation is incoherent, and reading one takes a minute.

Usage:
  coder dataset build [flags]
  coder dataset stats [--out DIR] [--json]
  coder dataset show <n> [--out DIR]

Flags:
  --source <list>         Comma-separated: coder, claude, codex, chat
  --file <path>           A chat JSONL to include. Repeatable
  --out <dir>             Where to write. Defaults to training/data
  --max-tokens <n>        Longest example to keep. Defaults to 4096
  --num-ctx <n>           Window to size tool results for
  --cwd <path>            Workspace the system prompt is built against
  --json                  Machine-readable output
  -h, --help              Display this help and exit

Examples:
  coder dataset build
  coder dataset build --source claude,codex --max-tokens 8192
  coder dataset stats
  coder dataset show 0
"""

UPDATE_HELP = """\
coder update
Install the newest release, or check whether there is one.

Releases are git tags. An update is a re-install of a named tag with
`uv tool install --force`, which is exactly what the installer runs -- so it
leaves you where a fresh install would, and nothing else on the machine moves.

A session also checks on its own, at most once a day, and says so in one line
of the greeting. That check asks the GitHub releases API for a version number
and sends nothing: no code, no prompts, no telemetry. It is skipped entirely
off a terminal and for a source checkout, and `update_check false` turns it
off for good.

Usage:
  coder update [--check]

Flags:
  --check                 Report what is available; install nothing
  -h, --help              Display this help and exit

Examples:
  coder update
      Install the newest release, if you are behind one

  coder update --check
      Say whether there is one, and stop there

  coder config set update_check false
      Never check for a release again
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
