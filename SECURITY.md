# Security

## Reporting a vulnerability

Please report security issues privately, through
[GitHub's private advisory form](https://github.com/thebkht/bkht-coder/security/advisories/new),
rather than as a public issue.

Include what you did, what happened, and — if you have one — a minimal repro.
You should get an acknowledgement within a week.

## Supported versions

There are no releases yet. The latest `main` is the only supported version, and
fixes land there.

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

**Remembered approvals are scoped to a call, not a tool.** Answering `a` at the
prompt remembers *that* command and *that* path — see `bkht/coder/rules.py`. It
does not hand the model a category of action.

## In scope

- Escaping the workspace root — writes or reads outside it that the policy
  should have refused
- Any mutating call reaching the system without approval in `ask` mode, or at
  all in `plan` mode
- A remembered approval matching more broadly than the call it was granted for
- **Prompt injection that bypasses the permission layer.** The agent reads files
  and command output it did not write, so content that tries to steer it is
  expected and is not itself a vulnerability. Content that gets a mutating call
  *past the approval prompt* is one, and is worth reporting.

## Out of scope

- The model proposing something wrong or destructive that you then approve.
  Reviewing the diff is what the prompt is for.
- Anything reached through `--auto`, which is documented as removing the gate.
- Vulnerabilities in Ollama, in the models themselves, or in `uv` — please
  report those upstream.
