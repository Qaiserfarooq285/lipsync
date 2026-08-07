#!/usr/bin/env bash
# Provision the isolated per-stage environments.
#
#   ./scripts/setup_envs.sh              # orchestrator + voice + lipsync
#   ./scripts/setup_envs.sh captions     # add the optional captions env
#   ./scripts/setup_envs.sh all          # everything
#
# Each GPU stage gets its own venv because their dependency pins are mutually
# exclusive (see requirements/*.txt). The CPU orchestrator drives them by
# subprocess; nothing heavy is ever installed into the orchestrator env.
#
# On Colab there is a single environment and no isolation is possible — install
# one stage's requirements per runtime instead. See notebooks/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WHICH="${1:-default}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- orchestrator -----------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  log "Creating orchestrator venv (.venv)"
  python3 -m venv .venv
fi
log "Installing orchestrator requirements"
.venv/bin/python -m pip install -q --upgrade pip setuptools wheel
.venv/bin/python -m pip install -q -r requirements.txt

UV=".venv/bin/uv"

make_env() {
  local name="$1" pyver="$2" reqs="$3"
  local dir=".venvs/$name"
  log "Creating $name env (Python $pyver)"
  "$UV" python install "$pyver" >/dev/null 2>&1 || true
  "$UV" venv --python "$pyver" "$dir"
  log "Installing $reqs into $dir (this pulls CUDA wheels; expect several GB)"
  VIRTUAL_ENV="$dir" "$UV" pip install --python "$dir/bin/python" -r "$reqs"
}

if [ "$WHICH" = "default" ] || [ "$WHICH" = "all" ] || [ "$WHICH" = "voice" ]; then
  make_env voice 3.12 requirements/voice.txt
fi

if [ "$WHICH" = "default" ] || [ "$WHICH" = "all" ] || [ "$WHICH" = "lipsync" ]; then
  make_env lipsync 3.10 requirements/lipsync.txt
  log "Vendoring LatentSync"
  ./scripts/vendor_latentsync.sh
fi

if [ "$WHICH" = "captions" ] || [ "$WHICH" = "all" ]; then
  make_env captions 3.12 requirements/captions.txt
fi

log "Done. Verify with: .venv/bin/python -m core.doctor"
