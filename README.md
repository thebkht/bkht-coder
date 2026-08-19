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
