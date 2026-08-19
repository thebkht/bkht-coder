#!/bin/sh
#
# One-line installer for bkht-coder.
#
#   curl -fsSL https://thebkht.com/install | sh
#
# Installs uv, Ollama and the model if they are missing, then puts `coder` on
# PATH. Every step is skipped when it is already satisfied, so re-running this
# is an upgrade rather than an error.
#
# Environment:
#   MODEL                   model tag to pull (default: picked from host RAM)
#   OLLAMA_HOST_URL         where the server should answer (default 127.0.0.1:11434)
#   BKHT_CODER_REF          git branch/tag to install instead of the default
#   BKHT_CODER_NO_MODEL=1   skip the model pull entirely
#   BKHT_CODER_YES=1        assume yes; required when there is no terminal
#   BKHT_CODER_ALLOW_ROOT=1 permit running as root on Linux
#
set -eu

REPO_URL="git+https://github.com/thebkht/bkht-coder.git"
MODEL_DEFAULT="qwen2.5-coder:14b"
MODEL_SMALL="qwen2.5-coder:7b"
HOST="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"

UV_INSTALLER="https://astral.sh/uv/install.sh"
OLLAMA_INSTALLER="https://ollama.com/install.sh"

pass() { printf '\033[1;32mPASS\033[0m %s\n' "$*"; }
step() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

server_up() { curl -sf -o /dev/null --max-time 2 "$HOST/api/tags"; }

wait_for_server() {
  tries="$1"
  while [ "$tries" -gt 0 ]; do
    if server_up; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

# ---------------------------------------------------------------- preflight

OS="$(uname -s)"
case "$OS" in
  Darwin|Linux) ;;
  *) fail "unsupported platform: $OS — see the Windows section of the README:
     https://github.com/thebkht/bkht-coder#install-manually--windows" ;;
esac

have curl || fail "curl not on PATH — install it and re-run"
have git  || fail "git not on PATH — install it and re-run"

if [ "$(id -u)" = "0" ] && [ "${BKHT_CODER_ALLOW_ROOT:-}" != "1" ]; then
  fail "running as root would install 'coder' where your user cannot reach it.
     Re-run as your normal user, or set BKHT_CODER_ALLOW_ROOT=1 to override."
fi

# ------------------------------------------------------------------ consent

# Pick the model tag now so the consent prompt can name it honestly.
detect_ram_gb() {
  case "$OS" in
    Darwin) bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
            echo $((bytes / 1024 / 1024 / 1024)) ;;
    Linux)  kb="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
            echo $((kb / 1024 / 1024)) ;;
  esac
}

RAM_GB="$(detect_ram_gb)"
if [ -n "${MODEL:-}" ]; then
  MODEL_TAG="$MODEL"
elif [ "$RAM_GB" -gt 0 ] && [ "$RAM_GB" -lt 12 ]; then
  MODEL_TAG="$MODEL_SMALL"
  warn "detected ${RAM_GB} GB of RAM — using $MODEL_SMALL instead of $MODEL_DEFAULT"
else
  MODEL_TAG="$MODEL_DEFAULT"
fi

# A server that already answers is the whole job done: don't install a second
# Ollama next to it. This is the WSL case -- the server runs on the Windows
# host, and the Linux side needs nothing but a URL.
SERVER_UP=0
if server_up; then SERVER_UP=1; fi

# The CLI reads OLLAMA_HOST, not OLLAMA_HOST_URL. Without this the `ollama`
# commands below would talk to localhost while everything else talks to $HOST.
OLLAMA_HOST="$HOST"
export OLLAMA_HOST

NEED_UV=0; have uv || NEED_UV=1
NEED_OLLAMA=0
if [ "$SERVER_UP" = 0 ] && ! have ollama; then NEED_OLLAMA=1; fi

printf '\nbkht-coder installer\n\n'
printf 'This will install:\n'
if [ "$NEED_UV" = 1 ]; then     printf '  * uv          (%s)\n' "$UV_INSTALLER"; fi
if [ "$NEED_OLLAMA" = 1 ]; then printf '  * Ollama      (%s)\n' "$OLLAMA_INSTALLER"; fi
if [ "${BKHT_CODER_NO_MODEL:-}" = "1" ]; then
  printf '  * the model   (skipped: BKHT_CODER_NO_MODEL=1)\n'
else
  printf '  * %s   (several GB, if not already pulled)\n' "$MODEL_TAG"
fi
printf '  * coder       (uv tool install %s)\n' "$REPO_URL"
if [ "$SERVER_UP" = 1 ]; then
  printf '\nAn Ollama server is already answering at %s — using it as is.\n' "$HOST"
fi
printf '\n'

if [ "${BKHT_CODER_YES:-}" = "1" ]; then
  step "BKHT_CODER_YES=1 — proceeding without asking"
else
  # Piped into sh, stdin is the script itself — ask the terminal directly.
  reply=""
  if [ -t 0 ]; then
    printf 'Continue? [y/N] '
    read -r reply || reply=""
  elif [ -e /dev/tty ] && (exec 3</dev/tty) 2>/dev/null; then
    printf 'Continue? [y/N] '
    read -r reply < /dev/tty || reply=""
  else
    fail "no terminal to confirm on. Re-run with BKHT_CODER_YES=1:
     curl -fsSL <url> | BKHT_CODER_YES=1 sh"
  fi
  case "$reply" in
    y|Y|yes|YES) ;;
    *) printf '\nAborted.\n'; exit 1 ;;
  esac
fi

# ----------------------------------------------------------------------- uv

if have uv; then
  pass "uv already installed ($(uv --version 2>/dev/null || echo unknown))"
else
  step "installing uv"
  curl -LsSf "$UV_INSTALLER" | sh || fail "uv install failed — see https://docs.astral.sh/uv/"
  # The installer drops uv in ~/.local/bin but only edits shell rc files, which
  # this process never re-reads.
  PATH="$HOME/.local/bin:$PATH"
  export PATH
  have uv || fail "uv installed but not on PATH — open a new shell and re-run"
  pass "uv installed"
fi

# --------------------------------------------------------- ollama and server

if [ "$SERVER_UP" = 1 ]; then
  pass "ollama already reachable at $HOST"
else
  if have ollama; then
    pass "ollama already installed"
  else
    step "installing ollama"
    case "$OS" in
      Darwin)
        if have brew; then
          brew install ollama || fail "brew install ollama failed"
        else
          fail "install Ollama from https://ollama.com/download (or install Homebrew first), then re-run"
        fi
        ;;
      Linux)
        curl -fsSL "$OLLAMA_INSTALLER" | sh || fail "ollama install failed — see https://ollama.com/download"
        ;;
    esac
    have ollama || fail "ollama installed but not on PATH — open a new shell and re-run"
    pass "ollama installed"
  fi

  # The Linux installer registers a systemd service; give it a moment to come
  # up on its own before starting a second copy.
  if [ "$OS" = "Linux" ] && have systemctl && systemctl is-enabled ollama >/dev/null 2>&1; then
    step "waiting for the ollama service"
    wait_for_server 15 || true
  fi
  if ! server_up; then
    step "starting ollama serve in the background"
    nohup ollama serve >/dev/null 2>&1 &
    wait_for_server 30 \
      || fail "ollama did not answer at $HOST after 30s.
     Start it yourself and re-run:  ollama serve"
  fi
  pass "ollama reachable at $HOST"
fi

# -------------------------------------------------------------------- model

# With no local CLI -- a remote server, typically the Windows host seen from
# WSL -- the same two operations go over the HTTP API instead.
model_present_api() {
  curl -sf --max-time 10 "$HOST/api/tags" | grep -Fq "\"name\":\"$MODEL_TAG\""
}

pull_model_api() {
  curl -fsS -N -X POST "$HOST/api/pull" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_TAG\"}" \
  | awk '
      match($0, /"status":"[^"]*"/) {
        s = substr($0, RSTART + 11, RLENGTH - 12)
        if (s != last) { print "    " s; fflush(); last = s }
        if (s == "success") ok = 1
      }
      /"error"/ { print "    " $0 > "/dev/stderr" }
      END { exit ok ? 0 : 1 }
    '
}

if [ "${BKHT_CODER_NO_MODEL:-}" = "1" ]; then
  warn "skipping the model pull (BKHT_CODER_NO_MODEL=1) — 'ollama pull $MODEL_TAG' before first use"
elif have ollama; then
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL_TAG"; then
    pass "model $MODEL_TAG already pulled"
  else
    step "pulling $MODEL_TAG — this is a multi-gigabyte download"
    ollama pull "$MODEL_TAG" \
      || fail "pull failed — retry with:  ollama pull $MODEL_TAG"
    pass "model $MODEL_TAG pulled"
  fi
elif model_present_api; then
  pass "model $MODEL_TAG already on the server at $HOST"
else
  step "pulling $MODEL_TAG on the server at $HOST — this is a multi-gigabyte download"
  pull_model_api \
    || fail "pull failed on $HOST — pull it there with:  ollama pull $MODEL_TAG"
  pass "model $MODEL_TAG pulled"
fi

# -------------------------------------------------------------------- coder

TARGET="$REPO_URL"
if [ -n "${BKHT_CODER_REF:-}" ]; then TARGET="$REPO_URL@$BKHT_CODER_REF"; fi

step "installing coder ($TARGET)"
uv tool install --force "$TARGET" || fail "uv tool install failed for $TARGET"
pass "coder installed"

if ! have coder; then
  # uv installs into its own bin dir; the rc-file edit needs a new shell.
  UV_BIN="${XDG_DATA_HOME:-$HOME/.local/share}/uv/tools"
  if [ -d "$HOME/.local/bin" ]; then UV_BIN="$HOME/.local/bin"; fi
  PATH="$UV_BIN:$PATH"
  export PATH
fi

if have coder; then
  coder --help >/dev/null 2>&1 || fail "coder is on PATH but 'coder --help' failed"
  pass "coder --help"
else
  warn "coder is installed but not on this shell's PATH."
  printf "     Run 'uv tool update-shell', or add this to your profile:\n"
  printf '       export PATH="$HOME/.local/bin:$PATH"\n'
fi

# --------------------------------------------------------------------- done

# coder defaults to localhost, so a server anywhere else has to be named on
# every run -- say so here rather than let the first session fail.
HOST_FLAG=""
case "$HOST" in
  http://localhost:11434|http://127.0.0.1:11434) ;;
  *) HOST_FLAG=" --host \"$HOST\"" ;;
esac

cat <<EOF

Done. Try:

  coder$HOST_FLAG                          # interactive REPL in the current directory
  coder$HOST_FLAG "add a --verbose flag"   # one-shot
  coder --help                   # every flag

Model:   $MODEL_TAG
Server:  $HOST
State:   ~/.bkht-coder/sessions/

EOF
