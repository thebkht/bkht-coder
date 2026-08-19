#!/usr/bin/env bash
#
# End-to-end check that coder can actually edit code via the local model.
# Mirrors ~/agent/scripts/verify.sh, but drives the live pytest suite.
#
#   ./scripts/verify.sh
#   MODEL=qwen2.5-coder:7b ./scripts/verify.sh
#
set -euo pipefail

MODEL="${MODEL:-qwen2.5-coder:14b}"
HOST="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"

pass() { printf '\033[1;32mPASS\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

cd "$(dirname "$0")/.."

command -v ollama >/dev/null 2>&1 || fail "ollama not on PATH"
pass "ollama found"

curl -sf -o /dev/null --max-time 5 "$HOST/api/tags" \
  || fail "ollama not reachable at $HOST — run 'ollama serve'"
pass "ollama reachable at $HOST"

ollama list | awk '{print $1}' | grep -qx "$MODEL" \
  || fail "model $MODEL not pulled — run 'ollama pull $MODEL'"
pass "model $MODEL present"

uv run pytest -q -m "not live" || fail "unit tests failed"
pass "unit tests"

uv run pytest -q -m live --model "$MODEL" || fail "live end-to-end failed"
pass "live end-to-end"
