#!/usr/bin/env sh
# Train the LoRA. Everything that decides anything is in lora.yaml.
set -eu
cd "$(dirname "$0")/.."

if [ ! -f training/data/train.jsonl ]; then
    echo "No dataset. Run: coder dataset build" >&2
    exit 1
fi

# Reported before the run rather than discovered during it: the run takes
# hours, and "how much am I training on" is the question worth answering first.
coder dataset stats || true

exec uv run --extra train mlx_lm.lora --config training/lora.yaml "$@"
