```
⣀⣸⣿⣿⣿⣿⣿⣿     bkht.coder
⣿⣿⠀⠀⠀⠀⣿⣿     A coding agent on a local Ollama.
⣿⣿⠀⠀⠀⠀⣿⣿     curl -fsSL https://thebkht.com/install.sh | sh
⣿⣿⣿⣿⣿⣿⡏⠉     Python 3.12+ · uv · Ollama · nothing leaves the machine
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
`/instructions`, `/skills`, `/jobs`, `/permissions`, `/model`, `/mode`,
`/doctor`, `/help`, `/exit`. `!cmd` shells out, and `exit` on its own leaves.

On a terminal the session streams: prose appears as the model writes it, a
status line shows elapsed time and tokens while it is quiet, and each tool call
is announced in words above the call itself. Approvals take a single key --
`y`, `n`, `a` to remember this call, or `d` to see the whole diff rather than
the first forty lines. Arrow keys recall earlier prompts, Tab completes slash
commands, and shift+tab cycles the permission mode -- which the line under the
prompt names as you type. Redirect the output and all of that goes away: piped
runs print the same plain transcript they always did.

State lives in `~/.bkht-coder/`: sessions under `sessions/`, prompt
history in `history`, remembered approvals in `permissions.json`, skills that
apply everywhere under `skills/`, background job logs under `jobs/`, slash
commands under `commands/`.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — used for the venv, the lockfile, and running the tests
- **[Ollama](https://ollama.com/download)**, serving on `http://localhost:11434`
- The model: `qwen2.5-coder:14b` (~9 GB on disk). About **16 GB of RAM** is
  enough at the default `num_ctx` of 16384, though turns are slower there;
  on 8 GB use `qwen2.5-coder:7b`.

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
irm https://thebkht.com/install.sh.ps1 | iex
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
| `BKHT_CODER_REF`        | Install a branch or tag instead of the default          |
| `BKHT_CODER_YES=1`      | Don't ask; required when there is no terminal to ask on |

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

There is no config file; everything is a flag, and the defaults are the ones
described above.

| Flag                | Default                  | What it does                                                     |
| ------------------- | ------------------------ | ---------------------------------------------------------------- |
| `--model`           | `qwen2.5-coder:14b`      | Ollama model tag                                                 |
| `--host`            | `http://localhost:11434` | Ollama server URL                                                |
| `--num-ctx`         | `16384`                  | Context window requested from Ollama (values ≤ 4096 are refused) |
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

`OLLAMA_HOST` is Ollama's own variable — set it before `ollama serve` to change
where the _server_ listens (e.g. `OLLAMA_HOST=0.0.0.0:11434`), then pass the
matching URL to `coder --host`.

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

An exact repeat of a tool call is refused rather than run (`agent.py`). Freeing
context necessarily costs the model some of what it read, and a model that has
lost a file reaches for it again — spending the window that made it forget, and
losing the file again. The refusal names what to do instead, and the model takes
it: in practice it switches to `offset`/`limit` and pages through.

And a turn that runs out of iterations or retries is asked for a final answer in
prose before it ends, so a bounded turn reports what it found instead of nothing.

If turns are still hitting the iteration cap, `--num-ctx` is the flag to raise.

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
