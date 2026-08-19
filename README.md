# bkht-coder

A coding agent built from scratch, running against a local Ollama server.

```sh
coder                          # interactive REPL in cwd
coder "add a --verbose flag"   # one-shot
coder --resume                 # continue last session here
coder --auto                   # no permission prompts
coder --plan                   # read-only
coder --model qwen2.5-coder:7b
coder review --base main       # review a branch
```

State lives in `~/.bkht-coder/sessions/`.

## Development

```sh
uv run pytest -q          # unit + loop tests, no model needed
uv run pytest -q -m live  # end-to-end against a running Ollama
```
