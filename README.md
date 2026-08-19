# bkht-coder

A coding agent built from scratch, running against a local Ollama server.

```sh
coder                          # interactive REPL in the current directory
coder "add a --verbose flag"   # one-shot
coder --resume                 # continue the last session here
coder --auto                   # no permission prompts
coder --plan                   # read-only
coder --model qwen2.5-coder:7b
```

Slash commands: `/tools`, `/context`, `/clear`, `/undo`, `/diff`, `/review`,
`/model`, `/mode`, `/help`, `/exit`. `!cmd` shells out.

State lives in `~/.bkht-coder/sessions/`.

## Requirements

* **Python 3.12+**
* **[uv](https://docs.astral.sh/uv/)** — used for the venv, the lockfile, and running the tests
* **[Ollama](https://ollama.com/download)**, serving on `http://localhost:11434`
* The model: `qwen2.5-coder:14b` (~9 GB on disk). About **16 GB of RAM** is
  enough at the default `num_ctx` of 8192; on 8 GB use `qwen2.5-coder:7b`.

## Install — Linux / macOS

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

## Install — Windows

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

Under WSL, Ollama running on the Windows host is not on WSL's `localhost` —
point the agent at the host explicitly:

```sh
coder --host "http://$(ip route show default | awk '{print $3}'):11434"
```

## Configuration

There is no config file; everything is a flag, and the defaults are the ones
described above.

| Flag | Default | What it does |
|---|---|---|
| `--model` | `qwen2.5-coder:14b` | Ollama model tag |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--num-ctx` | `8192` | Context window requested from Ollama (values ≤ 4096 are refused) |
| `--cwd` | `.` | Workspace root the tools are confined to |
| `--max-iterations` | `25` | Cap on agent loop iterations per task |
| `--no-instructions` | off | Ignore `AGENTS.md` / `CLAUDE.md` |

Two environment variables are read by the **tooling**, not the agent:

| Variable | Used by | Default |
|---|---|---|
| `OLLAMA_HOST_URL` | `scripts/verify.sh` | `http://127.0.0.1:11434` |
| `MODEL` | `scripts/verify.sh` (passed on as `pytest --model`) | `qwen2.5-coder:14b` |

`OLLAMA_HOST` is Ollama's own variable — set it before `ollama serve` to change
where the *server* listens (e.g. `OLLAMA_HOST=0.0.0.0:11434`), then pass the
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

## Checking the install

```sh
ollama list                       # the model tag should appear
uv run pytest -q -m "not live"    # no model needed
./scripts/verify.sh               # full preflight + live suite (bash)
```

Common failures:

* **`ollama not reachable`** — the server isn't running (`ollama serve`), or
  it's bound to another address; check with `curl http://localhost:11434/api/tags`.
* **Every turn takes minutes** — `num_ctx` is too large for available RAM; see
  the table under *How it talks to the model*. Lower it or use the 7b model.
* **The model ignores tools / replies with JSON text** — expected for this
  class of model, and handled; see the same section.

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

## How it talks to the model

`qwen2.5-coder:14b` emits tool calls as ordinary message **content** with
`message.tool_calls` left `null`. A conventional loop that checks `tool_calls`
sees nothing and halts. So calls are parsed out of content with a
brace-matching scan (`parsing.py`), and native `tool_calls` are accepted too
when present — `provider.py` normalizes both into one type.

`options.num_ctx` is always sent. Ollama defaults to 2048 and silently
truncates past it, which is the most common cause of a bad local-model session;
a `num_ctx` below 4096 is refused outright.

The default is **8192**, not the model's native 32768, because the binding
constraint is host RAM. Measured on a 16 GB machine with `qwen2.5-coder:14b`,
one warm trivial completion:

| `num_ctx` | Placement | Size | Warm turn |
|---|---|---|---|
| 8192 | 100% GPU | 10 GB | 0.9 s |
| 16384 | 9% CPU / 91% GPU | 12 GB | 11.1 s |
| 32768 | 27% CPU / 73% GPU | 15 GB | >300 s (timed out) |

Past 8192 the KV cache pushes the working set off the GPU and every turn pays
for it. On a machine with more memory, raise it with `--num-ctx` — that is the
only change needed. This is also why context compaction earns its place: an
8K window fills quickly.

## Development

```sh
uv run pytest -q                # unit + loop tests, no model needed
uv run pytest -q -m live        # end-to-end against a running Ollama
./scripts/verify.sh             # preflight checks, then both suites
```

The live suite includes an accuracy corpus (`tests/corpus/`) of diffs with
planted bugs and known-good code. It reports recall and precision, so a prompt
change can be judged rather than guessed at.
