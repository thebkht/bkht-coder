#!/usr/bin/env sh
# Serve the fused model on the network.
#
# This is the half that answers "train here, run from anywhere": the machine
# with the memory does the thinking, and every other device points `host` at
# it. On the serving machine:
#
#     training/serve.sh
#
# and from a laptop, a tablet, or a phone on the same network:
#
#     coder --host http://<this-machine>:8080
#     coder config set host http://<this-machine>:8080    # to keep it
#
# 0.0.0.0 binds every interface, which is the point and also the caution: the
# endpoint has no authentication. Run it on a network you trust, or put it
# behind a tunnel. Set CODER_API_KEY on both ends if the server in front of it
# checks one.
set -eu
cd "$(dirname "$0")/.."

MODEL="${1:-training/fused}"
PORT="${PORT:-8080}"

if [ ! -d "$MODEL" ]; then
    echo "No model at $MODEL. Run training/fuse.sh first." >&2
    exit 1
fi

echo "Serving $MODEL on http://0.0.0.0:$PORT"
echo "Check it with: coder doctor --provider local --host http://localhost:$PORT"

exec uv run --extra train mlx_lm.server \
    --model "$MODEL" --host 0.0.0.0 --port "$PORT"
