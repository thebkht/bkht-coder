#!/usr/bin/env sh
# Fold the adapter into the weights, producing a model a server can host.
#
# Fusing rather than serving the adapter separately: the fused model is one
# directory that `mlx_lm.server` loads with no other argument, which is what
# makes the serving step something you can run from a login shell without
# remembering what was trained against what.
set -eu
cd "$(dirname "$0")/.."

OUT="${1:-training/fused}"

exec uv run --extra train mlx_lm.fuse \
    --model "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit" \
    --adapter-path training/adapters \
    --save-path "$OUT"
