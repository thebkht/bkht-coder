# Security

## Reporting a vulnerability

Please report security issues privately, through
[GitHub's private advisory form](https://github.com/thebkht/bkht-coder/security/advisories/new),
rather than as a public issue.

Include what you did, what happened, and — if you have one — a minimal repro.
You should get an acknowledgement within a week.

## Supported versions

The latest release is the supported version, and fixes land on `main` first.

## What this tool actually does

`coder` is an agent that runs a language model's proposals on your machine: it
reads and writes files in the workspace and it executes shell commands. That is
the point of it, and it is also the whole of its risk. It is worth being
explicit about where the boundaries are.

**The model is not trusted.** `bkht/coder/permissions.py` is the policy layer —
the model proposes, the policy decides — and `bkht/coder/approval.py` is the
prompt you answer. Every mutating tool call goes through it.

**The three modes are a real difference in exposure.**

| Mode | Flag | What it means |
|---|---|---|
| `ask` | default | Every mutating call is shown and approved by hand |
| `auto` | `--auto` | No prompts. The model's proposals run as made |
| `plan` | `--plan` | Read-only; mutating calls are refused |

`--auto` removes the safety net deliberately. Don't point it at a workspace
whose contents you don't trust.

**Hooks execute arbitrary commands from a config file.** `config.json` may
carry a `hooks` block — commands fired before a tool call, after one, and at
the end of a turn (see the README) — and so may `agent/hooks/<event>/`, where
each executable file is one such command. They are your commands, run as you,
with your environment, and nothing prompts before one runs. Two things follow.
A workspace `.bkht-coder/config.json` or `agent/hooks/` you did not write is
code you are about to execute, so treat cloning a repository and starting
`coder` in it the way you would treat any other repo-supplied build script. And
`coder doctor` names every hook it can find, precisely so none of them is
invisible; `--no-hooks` runs a session with none of them.

A `pre_tool` hook can *refuse* a call, and that direction is a safety feature
rather than a risk — but it is a second line, not the first. The permission
layer is the first, and a hook is not a substitute for `ask` mode.

**Tools under `agent/tools/` are imported into the agent's own process.** They
are the workspace's Python, running before the first turn, with everything the
agent itself can reach — a larger hazard than a hook, which is at least a
command you typed into your own config file. Cloning a repository is never
enough to run one: the `agent/` directory has to carry an `agent.json` marker,
the `agent_tools` setting has to be on (it ships off), and `--no-agent-tools`
turns it off again for one session. `coder doctor` names every tool that would
load and the file it comes from, and a tool may not take a built-in's name —
one answering to `write_file` would take calls the permission layer had already
approved under that name.

**Remembered approvals are scoped to a call, not a tool.** Answering `a` at the
prompt remembers *that* command and *that* path — see `bkht/coder/rules.py`. It
does not hand the model a category of action.

## In scope

- Escaping the workspace root — writes or reads outside it that the policy
  should have refused
- Any mutating call reaching the system without approval in `ask` mode, or at
  all in `plan` mode
- A remembered approval matching more broadly than the call it was granted for
- A `pre_tool` hook that refuses a call, and the call running anyway
- **Prompt injection that bypasses the permission layer.** The agent reads files
  and command output it did not write, so content that tries to steer it is
  expected and is not itself a vulnerability. Content that gets a mutating call
  *past the approval prompt* is one, and is worth reporting.

## Out of scope

- The model proposing something wrong or destructive that you then approve.
  Reviewing the diff is what the prompt is for.
- Anything reached through `--auto`, which is documented as removing the gate.
- A hook you configured doing what you configured it to do. A `hooks` block is
  a list of commands to run; running them is the feature.
- Vulnerabilities in Ollama, in the models themselves, or in `uv` — please
  report those upstream.
