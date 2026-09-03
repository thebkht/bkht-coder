```
⣀⣸⣿⣿⣿⣿⣿⣿     bkht.coder
⣿⣿⠀⠀⠀⠀⣿⣿     A coding agent on a model server you control.
⣿⣿⠀⠀⠀⠀⣿⣿     curl -fsSL https://thebkht.com/install.sh | sh
⣿⣿⣿⣿⣿⣿⡏⠉     Python 3.12+ · uv · Ollama or MLX · nothing leaves the machine
```

```sh
coder                          # interactive REPL in the current directory
coder doctor                   # check this install can actually run a turn
coder "add a --verbose flag"   # one-shot
coder --resume                 # continue the last session here
coder --auto                   # no permission prompts
coder --plan                   # read-only
coder --no-scout               # don't search the workspace before each task
coder --model qwen2.5-coder:7b
```

Slash commands: `/tools`, `/context`, `/clear`, `/undo`, `/diff`, `/review`,
`/instructions`, `/skills`, `/jobs`, `/sessions`, `/permissions`, `/model`,
`/mode`, `/config`, `/doctor`, `/help`, `/exit`. `!cmd` shells out, and `exit`
on its own leaves.

On a terminal the session streams: prose appears as the model writes it, a
status line shows elapsed time and tokens while it is quiet, and each tool call
is announced in words above the call itself, with a count of what came back
under it. The prompt block stays on screen for the length of a turn, so where
you are and how full the window is are readable while the model works rather
than only between turns. Approvals take a single key — `y`, `n`, `a` to
remember this call, or `d` to see the whole diff rather than the first forty
lines. **Esc stops a running turn** — including during the long wait before
the first token — and so does Ctrl-C. Redirect the output and all of that goes
away: piped runs print the same plain transcript they always did.

The prompt takes more than one line. Paste a forty-line block and it arrives as
one prompt rather than forty; **alt+enter** or a trailing `\` opens a line, and
the arrows move within the buffer before they reach back into the history. A
paste longer than four lines folds to a single `[Pasted text #1, 230 lines]`
chip and is put back in full when the line is sent — numbered, so a prompt
carrying two of them can say which is which. **Ctrl-V** attaches an image
from the clipboard — terminals cannot deliver one in a paste, so it has a key of its
own — and says at once whether the model you are running can actually see it.
Tab completes slash commands, and shift+tab cycles the permission mode, which
the line under the prompt names as you type.

State lives in `~/.bkht-coder/`: sessions under `sessions/`, prompt
history in `history`, remembered approvals in `permissions.json`, skills that
apply everywhere under `skills/`, background job logs under `jobs/`, pasted
images under `images/`, slash commands under `commands/`, and persistent
settings in `config.json`.

Claude Code and Codex keep transcripts on the same machine, and `coder sessions
--agent all` lists theirs beside its own for the directory you are in;
`coder session claude/1be46299` reads one. Theirs are read-only — their agent
holds state this one has never seen, so there is nothing honest to resume.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — used for the venv, the lockfile, and running the tests
- **A model server.** Either one works, and the installer sets up the first:
  - **[Ollama](https://ollama.com/download)** on `http://localhost:11434`, with
    `qwen2.5-coder:14b` pulled. Nothing to configure.
  - **Anything speaking the OpenAI API** on `http://localhost:8080` —
    `mlx_lm.server`, `llama-server`, vLLM. This is the `local` backend, it is
    the default, and it is what serves a model you fine-tuned yourself. See
    [training/README.md](training/README.md).
- The model: `qwen2.5-coder:14b` (~9 GB on disk). About **16 GB of RAM** is
  enough at the default `num_ctx` of 16384, though turns are slower there;
  on 8 GB use `qwen2.5-coder:7b`.

The default is `local`, and **nothing breaks if you have only Ollama**: with
nothing serving on 8080 and Ollama up, the session runs on Ollama and says so
in one line. Only a backend you named yourself fails rather than falling back —
running a different model than the one you typed would be the worse answer.

`--host` is why the two are worth separating. It may name another machine, so
the box with the memory can serve while you drive from a laptop.

On a discrete GPU the number that binds is the card, not the machine. What has
to stay resident is the weights plus the KV cache, and at `num_ctx` 16384 that
is 8.4 + 3.0 GB for the 14b and 4.4 + 0.9 GB for the 7b — so **an 8 GB card
runs the 7b at the full window and cannot hold the 14b at any window**. There
is no trick that avoids this: decoding reads every weight once per token, so
whatever does not fit is not streamed in the background, it is walked over the
bus while you wait. `coder doctor` reads the card's memory, asks the server what
your model actually costs, and then reports the split Ollama chose:

```
  ok    num_ctx       16384 tokens, about 5 GB of 8 GB of VRAM
  ok    placement     100% on GPU (5.3 GB resident)
```

## Install

```sh
curl -fsSL https://thebkht.com/install.sh | sh
```

The same script, straight from the repo, if you'd rather not trust a redirect:

```sh
curl -fsSL https://raw.githubusercontent.com/thebkht/bkht-coder/main/scripts/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm https://thebkht.com/install.ps1 | iex
```

Installs uv, Ollama and the model if they are missing, then puts `coder` on
your `PATH`. It lists what it is about to install and asks once before
touching anything; re-running it upgrades an existing install.

If an Ollama server is already answering — including one on another machine,
via `OLLAMA_HOST_URL` — the installer uses it and installs no second copy. That
is the WSL case: point it at the Windows host and the Linux side gets nothing
but `coder`.

```sh
OLLAMA_HOST_URL="http://$(ip route show default | awk '{print $3}'):11434" \
  curl -fsSL https://thebkht.com/install.sh | sh
```

Read it first if you'd rather not pipe a script into a shell — it is
`scripts/install.sh`, and the manual steps it automates are right below.

| Variable                | What it does                                            |
| ----------------------- | ------------------------------------------------------- |
| `MODEL`                 | Model tag to pull, overriding the RAM-based choice      |
| `BKHT_CODER_NO_MODEL=1` | Skip the model pull; `ollama pull` it yourself later    |
| `BKHT_CODER_REF`        | Install a branch or tag instead of the newest release   |
| `BKHT_CODER_YES=1`      | Don't ask; required when there is no terminal to ask on |

## Updating

The installer takes the newest release tag, and so does the updater:

```sh
coder update            # install the newest release
coder update --check    # say whether there is one, and stop there
```

A session also checks on its own, at most once a day, and mentions it in one
line of the greeting:

```
v0.3.0 available · coder update
```

**That check is the only request coder makes that does not go to your own
Ollama.** It asks the GitHub releases API for a version number. It sends
nothing — no code, no prompts, no paths, no telemetry — and it runs in the
background off a cached answer, so it never delays a turn or fails one. The
line above the prompt is the whole of what it does with the answer; nothing is
ever installed without you typing `coder update`.

It is skipped entirely when the output is not a terminal, and when coder is
running from a source checkout. To switch it off for good:

```sh
coder config set update_check false
```

`coder doctor` reports the same thing, asked live rather than from the cache.

Releases are git tags — there is no package index in the middle, and an update
is a re-install of a named tag with the same `uv tool install` the installer
runs. [`CHANGELOG.md`](CHANGELOG.md) says what changed in each one.

## Install manually — Linux / macOS

```sh
# 1. uv
curl -LsSf https://astral.sh/uv/install.sh | sh      # or: brew install uv

# 2. Ollama
curl -fsSL https://ollama.com/install.sh | sh        # Linux
brew install ollama                                  # macOS

# 3. Start the server and pull the model
ollama serve &                                       # macOS app users: just launch Ollama
ollama pull qwen2.5-coder:14b

# 4. The agent
git clone https://github.com/thebkht/bkht-coder.git && cd bkht-coder
uv sync --extra dev                                  # creates .venv from uv.lock
uv run coder                                         # interactive REPL
```

To get `coder` on your `PATH` without the `uv run` prefix:

```sh
uv tool install --editable .        # then just: coder
# or, inside the project venv:
source .venv/bin/activate && coder
```

## Install manually — Windows

The steps `scripts/install.ps1` automates, if you'd rather run them yourself.
Use **PowerShell**. Everything works natively; only `scripts/verify.sh` needs a
bash (Git Bash or WSL).

```powershell
# 1. uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Ollama — download and run the installer from https://ollama.com/download
#    It installs a background service; no `ollama serve` needed.
ollama pull qwen2.5-coder:14b

# 3. The agent
git clone https://github.com/thebkht/bkht-coder.git; cd bkht-coder
uv sync --extra dev
uv run coder
```

Same as above for a bare `coder` command:

```powershell
uv tool install --editable .        # then just: coder
# or: .\.venv\Scripts\Activate.ps1 ; coder
```

On Windows, state lives in `%USERPROFILE%\.bkht-coder\sessions\`.

The shell tool works out of the box: with no `bash` on `PATH` the agent runs
commands through PowerShell and is told to write PowerShell syntax. Installing
[Git for Windows](https://git-scm.com/download/win) upgrades it to bash, which
the model writes more reliably — worth doing if you have the choice.

Under WSL, Ollama running on the Windows host is not on WSL's `localhost` —
point the agent at the host explicitly:

```sh
coder --host "http://$(ip route show default | awk '{print $3}'):11434"
```

## Configuration

Every knob is a flag, and every flag has a default. A preference you keep
retyping can be written down instead — see [Settings that
persist](#settings-that-persist) below — and a flag you actually type still
wins over anything on disk.

| Flag                | Default                  | What it does                                                     |
| ------------------- | ------------------------ | ---------------------------------------------------------------- |
| `--provider`        | `local`                  | Backend: `local`, `ollama`, `claude-code`, `codex`               |
| `--model`           | the backend's own        | Model tag, or the backend's own default                          |
| `--host`            | `http://localhost:8080`  | Model server URL; may name another machine                       |
| `--num-ctx`         | `16384`                  | Context window (values ≤ 4096 are refused). Requested from Ollama; on `local` the server fixes it and this is what coder plans for |
| `--temperature`     | `0.2`                    | Sampling temperature; low keeps tool calls well-formed           |
| `--cwd`             | `.`                      | Workspace root the tools are confined to                         |
| `--max-iterations`  | `25`                     | Cap on agent loop iterations per task                            |
| `--no-instructions` | off                      | Ignore `AGENTS.md` / `CLAUDE.md`                                 |
| `--no-skills`       | off                      | Ignore skills, and omit the `skill` tool                         |
| `--version`         | —                        | Print the version, and which copy of coder is running            |

A few environment variables are read by the **tooling**, not the agent:

| Variable                | Used by                                                                       | Default                  |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------ |
| `OLLAMA_HOST_URL`       | `scripts/verify.sh`, `scripts/install.sh`                                     | `http://127.0.0.1:11434` |
| `MODEL`                 | `scripts/verify.sh` (as `pytest --model`), `scripts/install.sh` (tag to pull) | `qwen2.5-coder:14b`      |
| `BKHT_CODER_NO_MODEL`   | `scripts/install.sh` — skip the model pull                                    | unset                    |
| `BKHT_CODER_REF`        | `scripts/install.sh` — branch or tag to install                               | unset                    |
| `BKHT_CODER_YES`        | `scripts/install.sh` — skip the confirmation prompt                           | unset                    |
| `BKHT_CODER_ALLOW_ROOT` | `scripts/install.sh` — permit running as root                                 | unset                    |

One is read by the agent itself: **`CODER_API_KEY`**, sent as a bearer token by
the `local` backend. Most local servers want no key; set it if you have put
something in front of yours that does. It is read from the environment rather
than the config file, because a config file is a thing people commit.

`OLLAMA_HOST` is Ollama's own variable — set it before `ollama serve` to change
where the _server_ listens (e.g. `OLLAMA_HOST=0.0.0.0:11434`), then pass the
matching URL to `coder --host`.

## Settings that persist

`coder config` writes the flags you would otherwise retype into a file:

```
coder config                              # every setting, and where it came from
coder config get model
coder config set model qwen2.5-coder:7b   # personal default, everywhere
coder config set --workspace num_ctx 8192 # this repo only
coder config unset model
coder config path                         # where the two files are
```

Two files, layered the same way skills and slash commands are: personal
defaults in `~/.bkht-coder/config.json`, and `<workspace>/.bkht-coder/config.json`
in the repo to override them for one project. A flag on the command line beats
both, so a written-down default never gets in the way of a one-off run.

The keys are `provider`, `model`, `host`, `num_ctx`, `temperature`, `mode`,
`scout`, `max_iterations`, `instructions`, `skills` and `update_check` — each of
the first ten an existing flag, so nothing becomes configurable that was not
already; `update_check` is the [release check](#updating), and the one setting
that governs a request leaving this machine. A bad file is not
fatal: the settings fall back to their defaults and the reason is printed once.

The same thing works mid-session: `/config` lists, `/config set <key> <value>`
writes and applies the change to the running agent where it can, and says so
plainly when it cannot — `provider`, `instructions` and `skills` wait for the
next session — as does `update_check`. Add `--workspace` to either to write the
repo's file instead.

## The four backends

Two keep the work on hardware you own:

| `provider`   | What it talks to                                    | Default host             |
| ------------ | --------------------------------------------------- | ------------------------ |
| `local`      | any OpenAI-compatible server — MLX, llama.cpp, vLLM | `http://localhost:8080`  |
| `ollama`     | Ollama's own API                                    | `http://localhost:11434` |
| `claude-code`| the `claude` command                                | —                        |
| `codex`      | the `codex` command                                 | —                        |

`local` is the default because it is the one that can serve a model you trained
yourself, and because its `host` may point at another machine. If nothing is
serving there and Ollama is, the session runs on Ollama and tells you.

### Borrowing a bigger model

The other two exist for when a task is past what a 14b local model can do, and
both use a login you already have rather than an API key:

```
coder config set provider claude-code    # the `claude` command
coder config set provider codex          # the `codex` command
coder --provider claude-code "..."       # or just for this run
```

Switching brings that backend's own model and window with it — `claude-code`
runs `opus` at 1M, `codex` runs `gpt-5.5` at 400k — unless you have pinned a
`model` yourself, in which case yours is kept and a wrong one fails loudly
rather than being quietly replaced.

**They are transports, not agents.** Claude Code and Codex are each a complete
coding agent, and none of that is wanted here: coder has a permission gate, a
snapshot store and `/undo`, and an edit made behind those is an edit you cannot
take back. So both are launched with their own tooling shut off — `--tools ""`
for Claude Code, a read-only sandbox for Codex, neither reading your settings
for them — and asked only to produce text. Every tool call in that text is
executed by coder, through the same prompt you would have seen locally.

Two things are true of both and of neither by accident. The work leaves the
machine, which is the thing the two local backends exist to avoid. And each turn
launches a process, because these tools keep no conversation between calls —
which costs a second and is otherwise free, since coder resends its whole
history every turn anyway.

`coder doctor` checks whichever backend is configured, and the checks differ
because the backends do: `ollama` gets the server, the pulled weights, this
machine's memory and where the weights actually landed; `local` gets the first
three, since the OpenAI API has no notion of where a model is resident and this
report does not invent numbers it cannot measure; the other two get a check
that the command is installed. Whether the login is still good is
not checked, because asking would spend your own quota to find out; a lapsed
login shows up on the first turn, in the tool's own words.

## Project instructions

Standing rules for a workspace go in a file, so they don't have to be repeated
in every prompt. Three are read, in this order, later ones winning:

1. `~/.bkht-coder/AGENTS.md` — applies everywhere
2. `<workspace>/AGENTS.md`
3. `<workspace>/CLAUDE.md`

Workspace root only: no parent-directory walk and no per-directory nesting.
Both cost context, and this model has little to spare.

```sh
echo "Use pytest, never unittest. Never edit files under generated/." > AGENTS.md
```

The text is inserted into the system prompt above the tool section, and the
model is told it overrides the general guidance but not the tool-call format.
Loaded files are named on startup — instructions that shape every answer
silently are worse than none. A total of 4000 characters is kept (2000 per
file); anything beyond that is cut, and the cut is announced in the prompt
rather than hidden.

In a session, `/instructions` shows what is loaded and `/instructions reload`
re-reads it, so editing the rules doesn't cost the conversation. `--resume`
picks up edits on its own, because the system prompt is rebuilt rather than
replayed.

## Long-running commands

The shell tool waits, and gives up at sixty seconds. That is right for the
commands an agent mostly runs — a test suite, a build, a `git log` — and useless
for the ones that are not supposed to finish. A dev server started through it
either ends the turn or is ended by it.

Those go through `background` instead, which returns as soon as the process is
up and hands back an id:

```
background(action="start", command="npm run dev")   -> started job 1
background(action="output", job_id="1")             -> the last 200 lines
background(action="stop", job_id="1")
background(action="list")
```

Output goes to a file under `~/.bkht-coder/jobs/`, not into the conversation. A
server that logs a line a second would otherwise fill an 8K window while nobody
was reading it — and when the model does ask, it gets the _end_ of the log,
because for a server the last thing it said is the only thing worth reading.

Four actions on one tool rather than four tools. The tool set is the scarce
resource on a 14b model, and one schema costs far less selection accuracy than
four names competing for it.

Jobs belong to the session and do not outlive it. Quitting stops them, and so
does crashing; a process the agent started, that the user never saw and cannot
easily find, is not something to leave running on their machine. `/jobs` lists
them and `/jobs stop <id>` ends one by hand.

## Remembered approvals

`a` at the approval prompt stores the decision and stops asking. What it stores
is **one exact call** -- this command, or this path, in this workspace -- and
never the tool it was made with. Approving `uv run pytest -q` does not approve
the next shell command, and approving a write to `src/api.py` does not approve a
write to anything else.

That distinction is the whole feature. A grant that quietly widens itself is
worse than no grant, because the user believes they know what they gave away.

Rules live in `~/.bkht-coder/permissions.json`, keyed by workspace, so one
project's approvals never fire in another, and they survive restarts -- a
session preference that has to be re-earned every morning is just a slower way
of typing `y`.

```sh
/permissions                                       # what is remembered here, with ids
/permissions remember allow bash {"command": "make test"}
/permissions remember deny bash {"command": "git push"}
/permissions revoke 4f2a9c11
```

`remember` stores a decision without running the call, which is the only way to
make one calmly -- mid-turn, with a diff on screen and a model waiting, is when
a user is least inclined to read what they are approving. Its arguments are
checked against the tool's own schema, because a rule that can never match looks
like a grant and behaves like nothing.

A denial reads to the model exactly like a fresh refusal. It is told not to
retry and nothing more: that a rule exists is not information it can use, and
telling it invites arguing with the rule.

`--auto` and `--plan` are unchanged and are decided before any rule is
consulted.

## Skills

`AGENTS.md` charges for its text on every turn, whether or not the turn has
anything to do with it. Even at `num_ctx` 16384 that is a real price — which is why
the instruction budget is capped at 4000 characters, and why long-form
procedure does not belong there.

A skill splits the cost in two. Its name and one-line description go into the
system prompt; the body is fetched with the `skill` tool only when the model
decides it applies. Twenty skills cost about what one paragraph of `AGENTS.md`
costs, and the twenty bodies cost nothing until one of them is the right one.

A skill is a directory with a `SKILL.md` in it:

```
.bkht-coder/skills/releasing/SKILL.md
```

```markdown
---
name: releasing
description: How to cut and publish a release of this project.
---

Bump the version in pyproject.toml, run the suite, tag it, then ...
```

Three roots are scanned, one level deep, later ones winning a name collision:

1. `~/.bkht-coder/skills/` — applies everywhere
2. `<workspace>/.claude/skills/` — read for compatibility with a workspace
   already set up for another agent; never written to
3. `<workspace>/.bkht-coder/skills/`

Files sitting beside a `SKILL.md` can be pulled in by name —
`skill(name="releasing", resource="checklist.md")` — and nothing outside the
skill's own directory can be, which is the same boundary the workspace root has
and the same check that draws it.

The `skill` tool is registered **only when a skill was actually found**. A
workspace without skills gets exactly the tool set it had before the feature
existed: an extra tool the model can never use successfully is not free, it is
one more wrong answer available at every step.

A skill missing its `name` or `description` is skipped and said so on startup.
Silently ignoring it would look identical to a skill the model simply chose not
to use, and the user would have no way to tell the two apart. `/skills` lists
what loaded, where each came from, and what was refused.

## The row under the prompt

```
                                          v0.3.0 available · coder update
────────────────────────────────────────────────────────────────────────
› add a --verbose flag
────────────────────────────────────────────────────────────────────────
 bkht-coder  (main)  ctx ██░░░░░░░░░░ 18% used  [qwen2.5-coder:7b]  5.2k tokens
▸▸ auto mode on (shift+tab to cycle)
```

Five things, and each is there because it changes what the next keystroke
means. The **directory** and the **branch** say which checkout is about to be
edited. The **meter** says how much of the window is left, and turns orange at
the point where the next turn is the one that will summarize — the thing that
used to happen without warning and take the model's record of what it had read
with it. The **model** and the **tokens** say what is answering and what the
session has spent.

They used to be in the greeting, which is scrollback: it went on saying `ask`
however many times Shift+Tab had been pressed, and `0 ctx` however many turns
had been taken. A fact that changes belongs where it can be redrawn, so the
greeting now says only what stays true.

A narrow terminal drops fields from the right rather than wrapping, because a
row that wrapped would put the caret arithmetic out by a line and smear the
block on every keystroke. What survives longest is the directory, the branch
and the meter.

A release waiting to be installed gets a row of its own, above the frame and
against the right edge — it is news about the next version of the program
rather than about the line being typed, and `update_check false` turns it off.

**The block stays while a turn runs.** Submitting used to take it off the
screen, so a turn ran against a bare spinner: the session lost its shape at
exactly the moment there was most to say about it. Now the spinner keeps the
frame pinned under itself and the answer scrolls above it. The input in it is
empty, and stays empty — this is a picture of the prompt, not an editor, and a
box with words in it would promise a turn could read what was typed into it.

It stays up while the answer is still arriving, too. Prose comes a fragment at
a time and almost never ends on a newline, so a block that stood down for an
unfinished sentence stood down for the whole answer. A half-written line keeps
its own row and the block sits below it, and the cursor is put back at the
column the sentence stopped at — modulo the terminal, because a sentence longer
than the screen has already wrapped and the count of what was written is not
the column it is in.

Where there is no line editor — Windows, a pipe, an IDE console — the same two
footer rows are printed above the input instead of under it. Above is the only
side `readline` leaves free, and the rows are worth more out of place than
absent.

## Searching before it answers

Every task is preceded by a keyword search of the workspace, made from the
words in the request itself, and the result is handed to the model as a tool
result it never had to ask for.

It exists because the model has `grep` and `glob`, is told to use them, and
often does not — it answers, or edits, from the flat file list in the system
prompt. That list is built once at startup and has nothing to do with what was
asked.

Terms come out of the message: quoted spans, paths, identifiers, and their
camelCase and snake_case parts, minus a stopword set. Files are then ranked by
where the terms land — a hit in the path, a line that looks like a definition,
and up to three mentions each — with a bonus for every term after the first, so
breadth of match beats one common word repeated. The top few files and their
best lines are rendered as `path:line: text`, the same shape `grep` returns,
inside a 2000-character budget.

A message with nothing to search for — "thanks", "yes go ahead" — is not
searched, and a search that fails is dropped rather than costing the turn. The
block says what it is, so the model treats it as a starting point rather than
the answer, and `--no-scout` turns it off. `/context` shows which it is.

## A plan it keeps, and somewhere else to read

Two settings, `planning` and `delegation`, both on. They are one answer to one
problem, approached from opposite ends.

The problem is the one the compaction section below describes. Reading is what
fills the window; most reading is spent finding out where something is rather
than on the thing itself; and the two mechanisms that free space — summarising
and eliding — work by throwing message history away. What goes first is the
model's own account of what it was doing.

Asked to review this codebase, a 14b read five files into its own window,
compacted, and stopped at the time cap still opening files, having answered
nothing about any of them. A 7b asked the same thing answered from the opening
keyword search and never opened a file at all. Different failures, same two
holes: nowhere to record a decision about what to do, and nowhere to do the
reading that is not the model's own context.

### `plan`

A short numbered list the model writes and ticks off.

```
● plan(steps=[read reviewer.py, read ci.py, write it up])
  0/3 done
  1. [ ] read reviewer.py
  2. [ ] read ci.py
  3. [ ] write it up
```

The list does not live in the conversation. It sits beside it, on the session,
and is appended to every single request — so it is still there after a
compaction has taken the messages that produced it, which is the whole reason
it exists. It is the last thing the model reads before it replies, the same
position and the same reasoning as the language reminder.

At most eight steps, one line each. A model that writes twelve has decomposed
the task instead of doing it, and every step is paid for on every subsequent
request. Rewriting the list drops the ticks: carrying them across a rewrite
means guessing which new step each old one became, and a wrong guess reports
work nobody did as done.

It is persisted with the session, so `--resume` resumes the plan too, and
`/clear` drops it — a plan that survived would meet the next turn with a
checklist for work nobody asked for. The plan is also the one tool result
printed to you in full rather than counted: it is four short lines, it is the
agent's own account of what it is doing, and a count would hide the single
thing worth watching, which is whether the list being ticked down is still the
list the model wrote.

`plan` survives into plan mode's read-only tool set, because producing a plan
is what plan mode is for.

### `task`

One self-contained question, handed to a second agent that searches and reads
on its own and hands back prose.

```
● task(instruction=summarise what review/ci.py does and who calls it)
  ● read_file(bkht/coder/review/ci.py)
    268 lines
  ● grep(from .ci import)
    4 matches in 3 files
  12 lines back
```

The sub-agent's reading is done in a window of its own and then thrown away.
The parent's history grows by one paragraph instead of by two files, and the
search that produced it is simply gone — which is correct, because the parent
never wanted the search.

Three bounds, all deliberate:

- **Read-only.** It cannot write files, run a shell, or start a job.
  Delegation exists to save context; a nested agent making changes you were
  never shown is a different feature with a different set of questions.
- **No nesting.** Its tool set has no `task`, so a turn cannot fan out into a
  tree of agents whose cost nobody bounded. It has no `plan` either, so it
  cannot rewrite the plan of the turn that delegated to it.
- **Its own clock** — three minutes and eight round trips, not the parent's
  ten minutes. A delegated search that runs the whole budget has spent the
  turn rather than saved it.

Its tool calls are shown to you as they happen, because a turn that goes quiet
for ninety seconds looks stuck. Its prose is not: that would stream into the
middle of the parent's answer. Esc reaches into it without anything arranging
that — the sub-agent runs on the thread the interrupt is raised on.

A sub-agent that stops without an answer is reported as a failed tool call
rather than as an empty success, so the parent goes and looks itself instead
of writing up an answer it never received.

### Turning them off

```sh
coder --no-planning               # this session
coder --no-delegation
coder config set planning false   # for good
coder config set delegation false
```

Both switches exist because of the argument the tool registry makes about
itself: the tool set is small on purpose, and every extra tool measurably costs
selection accuracy on a small model. That argument applies to these two as much
as to anything else, so they were measured rather than asserted. Across
twenty-four single-tool requests a nine-tool and an eleven-tool `qwen2.5-coder:14b`
chose identically every time; what the pair costs is about 347 tokens of system
prompt, on every request. The CHANGELOG has the numbers.

### One thing to know

`task` fires when [the opening workspace search](#searching-before-it-answers)
does not, and does not fire when it does. Asked to summarise a module and say
who calls it, a 14b under `--no-scout` delegates and answers from files a
sub-agent actually read; the same request with the search block present is
answered straight out of the search snippet, nothing opened.

That is the scout's own hazard rather than one these tools introduced — a 7b
does it without them — but it is what decides how often either tool is reached
for. `--no-scout` is the lever today, and making the two agree is the next
thing on the list.

## Checking the install

```sh
coder doctor                      # every check below, with the fix for each
coder doctor --json               # machine-readable; exits 1 if a check failed
```

It asks the questions this list used to ask by hand: which copy of coder is
running and where it was pointed, is the server answering, is the model pulled,
does `num_ctx` fit this machine's memory, is there a shell and a git, is
`~/.bkht-coder/` writable, and which instructions and skills loaded. Every
failure carries the command that fixes it — a check that reports a problem
without naming the fix has only moved the search, not ended it.

The first two checks answer the failures that do not look like failures.
`uv tool install` copies the package into an environment of its own, so the
`coder` on your `PATH` keeps running the version it was installed at while the
checkout moves on without it — the symptom is a feature that plainly exists in
the source and is missing from the program, and nothing else in the report
would explain it. `coder --version` prints the same thing on one line:

```sh
coder --version                   # coder 0.1.0 (/path/it/is/running/from)
uv tool install --force --editable .   # make the installed copy follow a checkout
```

The workspace check catches the other one: started in a home directory rather
than a project, nothing is broken, but every search has the whole of it to walk
and every `.claude/skills` directory under it loads at once.

The `num_ctx` check is the one worth having. It is fitted to the measured table
under _How it talks to the model_, and it warns when the context asked for
would push the KV cache off the GPU, which is what is actually happening when
every turn suddenly takes minutes. It names a size that fits rather than
telling you to experiment.

The rest of the suite:

```sh
ollama list                       # the model tag should appear
uv run pytest -q -m "not live"    # no model needed
./scripts/verify.sh               # full preflight + live suite (bash)
```

One failure `doctor` deliberately does not check for: **the model ignores tools
and replies with JSON text**. That is expected for this class of model, and it
is handled; see _How it talks to the model_.

## Your own slash commands

A prompt retyped every day belongs in a file. Drop a Markdown file in
`.bkht-coder/commands/` and its name becomes a command:

```
.bkht-coder/commands/audit.md   ->  /audit provider.py
```

```markdown
---
description: Look for swallowed errors.
---

Audit the error handling in $ARGUMENTS. Report only silent failures.
```

The body becomes the task. `$ARGUMENTS` is substituted where the file asks for
it and appended where it does not, so a file written without a placeholder
still takes arguments rather than discarding them. Frontmatter is optional and
only its `description` is read, which is what `/help` lists them with.

Files in `~/.bkht-coder/commands/` apply everywhere, and a workspace can shadow
one with its own. Nothing here can shadow a built-in: `/undo` has to keep
meaning `/undo`, and file lookup happens only after the built-in table has been
checked. Nor is any of it executable — the body is prose sent to the model, and
a slash command that could run something would be a permission gate with a back
door in it.

## Training a model of your own

The default backend exists to serve one. `coder dataset` builds a training
corpus out of the transcripts already on this machine — its own sessions, and
Claude Code's and Codex's — by translating every call into coder's protocol:

```
coder dataset build      # collect, translate, and report what was found
coder dataset show 0     # read one example as the model will read it
coder dataset stats
```

Translation is the substance. A model trained on `Read(file_path=…)` learns to
emit a call coder has no tool for; mapped to `read_file(path=…)` it learns the
one that exists. Arguments coder's tools do not accept are stripped, because an
unknown argument is a hard error at runtime rather than a warning. Both foreign
agents call tools in parallel and narrate before acting — coder's protocol
forbids both — so results are moved behind the call they answer and the
narration is dropped. Every rendered call is parsed back through coder's own
parser before it is written, because a fine-tune whose calls coder cannot read
is worse than none: it looks fluent, the loop corrects every reply, and no turn
ever finishes.

Read the histogram `build` prints, and read one example. If there is not much
data — the usual finding — `training/generate.py` runs real tasks through
`--provider claude-code`, which produces coder sessions with a frontier model
choosing the calls, so nothing needs translating afterwards.

Then [training/README.md](training/README.md): a LoRA on a 4-bit 14b that fits
16 GB, fused into one directory, served on `0.0.0.0:8080` for every device you
own.

## Uninstalling

```sh
uv tool uninstall bkht-coder     # removes the `coder` command
rm -rf ~/.bkht-coder             # sessions and global AGENTS.md
```

uv, Ollama and the pulled model are left alone. The installer will use them if
they are already there, so it doesn't assume it owns them — remove those the
way you'd remove anything else you installed.

## Code review

```sh
coder review                   # uncommitted changes (staged + unstaged) vs HEAD
coder review --staged          # staged only
coder review --base main       # everything on this branch, from the merge base
coder review HEAD~3..HEAD      # an explicit commit range
coder review --files src/api.py    # whole files, not a diff
coder review --json            # machine-readable, for CI (exits 1 on findings)
coder review --output report.md    # save a Markdown report
coder review --fix             # after reporting, offer to fix findings
```

Review runs one pass per dimension — correctness, error-handling, security,
tests — then a second, independent pass over every candidate whose job is to
**refute** it. Only findings that survive are reported. That pass is the point:
a 14b model produces a lot of confident, wrong findings, and an unfiltered
report is worse than no report because it teaches you to ignore it.

Review is read-only. `--fix` is a separate phase that goes through the normal
agent loop and the normal permission gate.

### In CI

```sh
coder review --base main --ci     # detected anyway; the flag forces it
coder review --ci off             # ...and this suppresses it
```

Inside a pipeline the report is in the wrong place. Progress is written for a
person watching a terminal, so a log file gets escape codes; and the findings
land in stdout, which is not where anyone reviewing the change is looking. `--ci`
fixes both. Progress becomes flat, collapsible log sections, and each finding is
handed to the platform in the form it renders **on the line of the diff it is
about**:

- **GitHub Actions** — `::error`/`::warning`/`::notice` annotations on the pull
  request, and the full Markdown report appended to the job summary.
- **GitLab CI** — a `gl-code-quality-report.json` for the merge request widget,
  which also annotates the diff. Point it elsewhere with `--code-quality`.

It is detected from `GITHUB_ACTIONS`, `GITLAB_CI`, or a plain `CI`, so nothing
needs configuring on either side; `--ci github` and `--ci gitlab` force a shape,
and `--ci off` gets the interactive output back.

Two behaviours change under `--ci`. A finding exits 1, so the job fails on one
rather than reporting into a log nobody reads. And `--fix` is skipped, because it
asks which findings to fix and CI cannot answer.

There is a workflow for each platform in this repository — `.github/workflows/review.yml`
and `.gitlab-ci.yml`. Both need a runner with Ollama and the model on it, so
both are off by default: the GitHub one waits for a `coder-review` label, and
the GitLab one is tagged `ollama`.

Both also refuse to run for a pull request from a fork, and that restriction is
the important one. A self-hosted runner is not ephemeral, and this job executes
the branch it checks out — `uv sync` runs the build, `coder review` runs this
package. From a fork that is a stranger's code on your own machine. A label does
not fence it: once a maintainer applies one, every later push to the same branch
re-runs the job with the label still attached. So the branch has to live in the
repository, which takes write access to create.

## Checking what an edit means

`edit_file` used to check one thing: that `old_string` appeared exactly once.
It wrote the bytes and reported "Edited", and every one of those statements was
true of `from .session import STATE_DIR, Input` — a name `session.py` has never
defined. The string matched. The write succeeded. The package stopped importing,
and nobody found out until the next command.

A local model invents a name because the sentence reads well. That is the
failure mode, not an occasional slip, so `verify.py` parses the result of every
Python write before it lands. Two checks, deliberately asymmetric:

- **A syntax error is refused.** Checked before the write, so there is nothing
  to roll back. A file that does not parse is never what anyone meant.
- **An unresolved import is reported.** Names can arrive at runtime — a star
  import, a conditional definition, a module `__getattr__` — so this warns
  rather than blocks. A check that stops a correct edit is worse than one that
  misses an incorrect one.

The warning goes to both readers. The model gets it appended to the tool result,
in the same breath as the success, while it can still fix it. The human gets it
under the diff at the approval prompt:

```
--- bkht/coder/commands.py
+++ bkht/coder/commands.py
@@ -18,5 +18,5 @@
-from .session import STATE_DIR
+from .session import STATE_DIR, Input

! bkht/coder/session.py does not define `Input`, imported on line 20.
Allow edit_file? [y] yes  [n] no  [a] always this call  [d] full diff
```

That is the moment someone is already deciding, and that line is the whole of
what they need in order to say no.

Both checks are static. The check that would catch everything is "does the
module still import?", and running it means executing the model's new code to
find out whether the model's new code is safe to execute.

Only relative imports resolving inside the workspace are checked, and the
name collection is deliberately over-inclusive — it currently reports nothing
across this repository's own source, which is the bar. One false alarm is enough
to teach everybody to ignore the next true one.

### Running the tests you already run

The paragraph above declines to execute the model's new code. That refusal is
about the module just written — but *your* test command is a different
proposition. It is a command you chose, wrote down, and already run by hand;
the agent running it is not the agent deciding what to run.

```sh
coder config set verify_command "pytest -q"     # nothing runs until you do this
```

After that, a turn that changed a file runs it before it answers, and a failure
goes back as a tool result the model corrects from — the same path a malformed
tool call takes:

```
● write_file(mathy.py)
  Wrote 6 lines
● running python3 -m pytest -q
● python3 -m pytest -q failed (exit 1)
● edit_file(mathy.py)
  Edited
● running python3 -m pytest -q
● python3 -m pytest -q passed
```

The details that keep it cheap:

- **It runs once the model says it is finished**, not after every write. A
  check inside the edit loop puts the test runner in the iteration budget; this
  costs one run for a turn that edited, and nothing at all for a turn that only
  read — which is most of them.
- **Twice per turn at most.** The first run is the check, the second is the fix
  being checked. A third would mean handing back a failure the model has already
  failed to fix once, so instead the second failure asks for an account: say
  what is still broken and stop. An unfinished fix that names the problem is
  worth more than a third attempt that runs out of iterations mid-edit.
- **A timeout is not a failure.** Nothing about "it did not finish in 120s"
  tells the model what to change, so it ends the turn and is reported to you
  rather than fed back.
- **Esc does not reach it.** The interrupt is a flag the main thread reads
  between bytecodes, and this thread is blocked in `waitpid` for the whole run.
  That is why the timeout is 120 seconds rather than generous: the bound on how
  long you wait here is the timeout, not the key. A suite slower than that
  wants a narrower command — one package, one file.

Nothing is ever inferred into `verify_command`. `coder doctor` will *suggest*
one from what it sees in the directory, and `--verify-command` sets it for a
single run, but the whole case for running anything here rests on the command
being yours:

```
  ok    verify        not set; no test command runs after an edit
        This project looks like `pytest -q` -- `coder config set verify_command 'pytest -q'` to check edits with it.
```

`--no-verify` turns it off for a session without erasing what you configured,
and `/context` says which it is.

## How it talks to the model

`qwen2.5-coder:14b` emits tool calls as ordinary message **content** with
`message.tool_calls` left `null`. A conventional loop that checks `tool_calls`
sees nothing and halts. So calls are parsed out of content with a
brace-matching scan (`parsing.py`), and native `tool_calls` are accepted too
when present — `provider.py` normalizes both into one type.

The `tools` array is deliberately **not** sent, even though `ollama show` lists
`tools` under Capabilities. Passing it makes Ollama render qwen2.5's own
`<tool_call></tool_call>` protocol into the prompt, contradicting the one this
system prompt states — and the model then honours neither reliably. Asked with
`tools` set, it answered with the plain JSON object three times out of three and
left `message.tool_calls` null every time. Content is the transport that works;
the native path stays in `collect()` for a model that keeps the promise.

`temperature` defaults to **0.2**. Ollama's own default of 0.8 is tuned for
prose, and every tool call here is a JSON object that has to be exactly right.
Not 0.0: a model that has taken a wrong turn repeats it verbatim on every
retry, and the retry exists to get a different answer.

`keep_alive` is **30m**, so a conversation does not reload nine gigabytes of
weights between two turns on Ollama's five-minute idle timer.

The `local` backend speaks the same protocol over a different wire, and three
things differ in ways a naive port gets wrong. The stream is SSE, so every
payload arrives behind `data: ` and the stream ends with a literal `[DONE]`
that is not JSON. Native tool calls arrive in pieces — the `arguments` string
split across as many deltas as the server felt like, keyed by `index` — so a
call is only complete at the end of the stream. And usage is omitted entirely
unless `stream_options.include_usage` is sent, which is what the context meter
counts with.

`num_ctx` means something different there, too. Ollama takes it as a request;
an OpenAI-compatible server fixes the window when it starts, so the number is
what coder *plans* for. If the two disagree the server wins and the meter is
what is wrong.

**The reply is read on a thread of its own**, and handed to the renderer over a
queue. Two things follow from that, and the second was the reason for it.

Esc raises `KeyboardInterrupt` in the main thread, and the main thread only
notices between bytecodes. Blocked in a socket read it is running no bytecode
at all — so an Esc pressed during the wait before the first token was
remembered and delivered whenever the model happened to speak, which on a 14b
is most of a minute later. Waiting on a queue in short hops is bytecode, so the
key now lands when it is pressed.

It does **not** make a turn faster, which is worth writing down because it
looks as though it should: the socket no longer goes undrained while the screen
is being written to, and one thread used to do both — read a chunk, parse and
draw it, and only then go back for the next. Measured over six alternating
turns per version, same prompt, warm weights, `qwen2.5-coder:7b`:

| Version | Median | Mean  | Range       |
| ------- | ------ | ----- | ----------- |
| v0.2.0  | 22.8   | 22.9  | 22.1 – 23.6 |
| v0.3.0  | 22.9   | 22.4  | 19.1 – 24.0 |

Tokens per second, and a 0.6% difference in medians inside overlapping ranges,
which is no difference. What bounds a turn is the GPU generating tokens, not
the client collecting them, and draining a socket sooner cannot help when there
is nothing waiting in it. The turn *feels* quicker because the block stays on
screen throughout, which is a different claim and the one actually being made.

`options.num_ctx` is always sent. Ollama defaults to 2048 and silently
truncates past it, which is the most common cause of a bad local-model session;
a `num_ctx` below 4096 is refused outright.

The default is **16384**, not the model's native 32768, because the binding
constraint is host RAM. Measured on a 16 GB machine with `qwen2.5-coder:14b`,
one warm trivial completion:

| `num_ctx` | Placement         | Size  | Warm turn          |
| --------- | ----------------- | ----- | ------------------ |
| 8192      | 100% GPU          | 10 GB | 0.9 s              |
| 16384     | 9% CPU / 91% GPU  | 12 GB | 11.1 s             |
| 32768     | 27% CPU / 73% GPU | 15 GB | >300 s (timed out) |

8192 is the fastest number in that table and the wrong default. The table
measures one trivial completion; a real turn is a conversation, and at 8192 it
cannot hold a source file and think at the same time — this project's own
`cli.py` is ~6,900 tokens, 85% of the window. The turn does not fail loudly: it
reads a file, frees context to make room, loses the file, and reads it again,
spending its whole iteration budget paging. On the task that exposed this, 8192
gave 25 iterations and no answer; 16384 gave eight tool calls and a complete one.

So the default pays about ten seconds a turn to be able to finish. On a machine
with less memory, drop to `--num-ctx 8192` — `coder doctor` says when to, and
names what it costs.

An 8K window fills quickly, so several things are sized against it rather than fixed.
Tool output is capped at a quarter of the window (`tools/base.py`), because a
683-line file is 85% of 8192 tokens and a single `read_file` used to be able to
fill the context on its own. And a turn summarizes **once**: after that,
pressure is relieved by eliding older tool output, which is free. Summarizing
repeatedly was worse than not summarizing at all — each pass cost a full model
call and threw away the model's record of what it had already read, so it read
it again, and the turn spun until the iteration cap.

An exact repeat of a tool call is not run again (`agent.py`). Freeing context
necessarily costs the model some of what it read, and a model that has lost a
file reaches for it again — spending the window that made it forget, and losing
the file again. What comes back instead is **the result the call returned the
first time**, replayed out of the history a few messages up.

Handing it back is the whole point, and it took a bad session to see why. A
refusal that returned nothing left the model owing an answer it had just been
told it could not have, and a model in that position writes down what it
remembers: it reported the contents of a file it had never been shown, in the
confident register of a real reading. Replaying costs nothing and removes the
reason to invent. Only calls that actually ran are replayed — one refused
permission never happened.

Every bound leaves through the same door: a turn that runs out is asked for a
final answer in prose, so it reports what it found instead of nothing. There
are five ways to run out.

| `stopped`       | What it means                                       |
| --------------- | --------------------------------------------------- |
| `answered`      | The model stopped calling tools and wrote prose      |
| `iteration-cap` | 25 round trips                                       |
| `retry-cap`     | Three rounds in a row where nothing worked           |
| `looping`       | Three calls it had already made, answered each time  |
| `time-cap`      | Ten minutes                                          |

`looping` and `time-cap` are the newer two, and both come from one bad session.
Nothing had bounded the clock: the iteration cap counts round trips and the
retry cap counts only rounds where *every* call failed, so a turn making
distinct, successful calls that went nowhere was bounded by neither. That one
ran 1180 seconds and answered nothing.

If turns are still hitting the iteration cap, `--num-ctx` is the flag to raise.

There is a `gh` tool now, and a `glab` one, each registered only when its CLI
is on the PATH. They read: a run and its logs, a pull or merge request and its
diff, an issue. Anything that would write — `pr merge`, `run rerun`, an `api`
call with `--method POST` — is refused by the tool rather than left to the
permission gate, because merging a pull request is not a thing to approve one
keypress at a time in the middle of a turn that was only supposed to read a
log. The shell tool is still there, behind the gate, for when somebody means
it. The command is split into an argument list and run without a shell, so a
`;` in it is a `;`.

They exist because of the same session. `gh` was installed and logged in the
whole time, and the model wrote a `curl` with a placeholder token instead —
nothing had told it the capability was there. Naming a capability is most of
what makes it get used.

A command containing an obvious placeholder — `YOUR_GITHUB_TOKEN`, `<your
token>` — is refused before it runs. A model without a credential writes the
sentence that usually surrounds one, and the shell runs it: the request in that
same session authenticated as nobody, and the turn spent what was left of
itself diagnosing a credentials problem it had invented. The rule is narrow,
because redirection and markup in a `grep` have to keep working.

And the workspace search that opens each turn is skipped when the request names
something the workspace does not contain — a URL, an issue number, a run id.
Asked about GitHub run 33185669396 it matched `.github/` and `review/`, both
real directories, and pointed the turn at the workflow file, which says what
the job would do and nothing at all about what it did.

## Development

```sh
uv run pytest -q                # unit + loop tests, no model needed
uv run pytest -q -m live        # end-to-end against a running Ollama
./scripts/verify.sh             # preflight checks, then both suites
```

The live suite includes an accuracy corpus (`tests/corpus/`) of diffs with
planted bugs and known-good code. It reports recall and precision, so a prompt
change can be judged rather than guessed at.

## Contributing

Issues and pull requests are welcome — `CONTRIBUTING.md` covers the setup, the
two test suites, and how a prompt change gets judged against the corpus rather
than guessed at. Security issues go through a
[private advisory](https://github.com/thebkht/bkht-coder/security/advisories/new)
instead; `SECURITY.md` says what is in scope and what `--auto` gives up.

## License

MIT. See `LICENSE`.
