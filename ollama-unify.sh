#!/usr/bin/env bash
# ollama-unify — consolidate scattered ollama model stores into one canonical location
# https://github.com/robit-man/ollama-unify
#
# Detects every ollama model directory referenced by:
#   - $HOME/.ollama/models (per-user default)
#   - /usr/share/ollama/.ollama/models (system-user default)
#   - /etc/default/ollama, /etc/environment, systemd unit env (OLLAMA_MODELS)
#   - Live `ollama runner` cmdlines
# Then interactively unifies them at a destination of your choosing, picks the
# fastest available transfer method (same-fs mv / reflink / NVMe rsync), and
# optionally rewires systemd + shell rc + backward-compat symlinks.
#
# Safety: never deletes data. Renames originals to .bak / .orphan-blobs for you
# to remove after verifying.

set -euo pipefail

# ───────────────────────────────────────────────────────────────────── colors
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'; C_GRN=$'\033[32m'
  C_YEL=$'\033[33m'; C_BLU=$'\033[34m'; C_CYN=$'\033[36m'; C_RST=$'\033[0m'
else
  C_DIM=; C_BOLD=; C_RED=; C_GRN=; C_YEL=; C_BLU=; C_CYN=; C_RST=
fi
say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n%s%s%s\n' "$C_BOLD$C_CYN" "$*" "$C_RST"; }
ok()   { printf '%s✓%s %s\n' "$C_GRN" "$C_RST" "$*"; }
warn() { printf '%s!%s %s\n' "$C_YEL" "$C_RST" "$*" >&2; }
err()  { printf '%s✗%s %s\n' "$C_RED" "$C_RST" "$*" >&2; }
ask()  { local prompt="$1" default="${2:-}" reply; if [ -n "$default" ]; then
           read -r -p "$prompt [$default]: " reply < /dev/tty || true
           printf '%s' "${reply:-$default}"
         else
           read -r -p "$prompt: " reply < /dev/tty || true
           printf '%s' "$reply"
         fi; }
confirm() { local reply; reply=$(ask "$1" "${2:-N}"); case "${reply,,}" in y|yes|true|1) return 0 ;; *) return 1 ;; esac; }

# ───────────────────────────────────────────────────────────────────── banner
banner() {
  cat <<'BANNER'
  ___  _ _                                       _  __
 / _ \| | | __ _ _ __ ___   __ _    _   _ _ __ (_)/ _|_   _
| | | | | |/ _` | '_ ` _ \ / _` |  | | | | '_ \| | |_| | | |
| |_| | | | (_| | | | | | | (_| |  | |_| | | | | |  _| |_| |
 \___/|_|_|\__,_|_| |_| |_|\__,_|   \__,_|_| |_|_|_|  \__, |
                                                      |___/
BANNER
  printf '%sUnify scattered Ollama model stores into one canonical location.%s\n\n' "$C_DIM" "$C_RST"
}

# ───────────────────────────────────────────────────── prerequisite checks
require() { command -v "$1" >/dev/null 2>&1 || { err "missing required command: $1"; exit 2; }; }
require rsync; require du; require df; require find; require stat; require awk; require sort

HAS_SUDO=0; command -v sudo >/dev/null 2>&1 && HAS_SUDO=1
HAS_SYSTEMD=0; command -v systemctl >/dev/null 2>&1 && systemctl --version >/dev/null 2>&1 && HAS_SYSTEMD=1
HAS_JQ=0; command -v jq >/dev/null 2>&1 && HAS_JQ=1
HAS_CURL=0; command -v curl >/dev/null 2>&1 && HAS_CURL=1
[ "$(uname -s)" = "Linux" ] || warn "Tested on Linux only. macOS/WSL may need manual tweaks."

# ─────────────────────────────────────────── store discovery (read-only)
declare -a STORE_PATHS=()
add_store() {
  local p="$1"
  [ -z "$p" ] && return
  # canonicalize without requiring existence
  p="$(readlink -m -- "$p" 2>/dev/null || echo "$p")"
  [ -d "$p" ] || return
  for existing in "${STORE_PATHS[@]:-}"; do [ "$existing" = "$p" ] && return; done
  STORE_PATHS+=("$p")
}

discover_stores() {
  hdr "Scanning for ollama model stores…"
  # default locations
  add_store "${HOME}/.ollama/models"
  add_store "/usr/share/ollama/.ollama/models"
  add_store "/var/lib/ollama/.ollama/models"
  add_store "/root/.ollama/models"
  # shell env
  [ -n "${OLLAMA_MODELS:-}" ] && add_store "$OLLAMA_MODELS"
  # /etc/default/ollama
  if [ -r /etc/default/ollama ]; then
    local v
    v=$(grep -E '^OLLAMA_MODELS=' /etc/default/ollama | tail -1 | cut -d= -f2- | tr -d '"'"'"'')
    add_store "$v"
  fi
  # /etc/environment
  if [ -r /etc/environment ]; then
    local v
    v=$(grep -E '^OLLAMA_MODELS=' /etc/environment | tail -1 | cut -d= -f2- | tr -d '"'"'"'')
    add_store "$v"
  fi
  # systemd unit + drop-ins
  if [ "$HAS_SYSTEMD" = 1 ]; then
    if systemctl cat ollama >/dev/null 2>&1; then
      local v
      v=$(systemctl cat ollama 2>/dev/null | grep -oP 'OLLAMA_MODELS=\K[^"\s]+' | tail -1)
      add_store "$v"
    fi
  fi
  # running runner cmdlines
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r path; do add_store "$path"; done < <(
      pgrep -af 'ollama runner' 2>/dev/null \
        | grep -oP '/[^ ]+/blobs/sha256-[a-f0-9]+' \
        | sed 's|/blobs/.*||' | sort -u
    )
  fi
}

count_manifests() { find "$1/manifests" -type f 2>/dev/null | wc -l; }
count_blobs()     { find "$1/blobs"     -type f 2>/dev/null | wc -l; }
fs_of()           { df --output=source,target "$1" 2>/dev/null | tail -1 | awk '{print $1" ("$2")"}'; }
dev_of()          { df --output=source "$1" 2>/dev/null | tail -1 | tr -d ' '; }
mount_of()        { df --output=target "$1" 2>/dev/null | tail -1 | tr -d ' '; }
own_of()          { stat -c '%U:%G' "$1" 2>/dev/null || echo "?"; }
size_of()         { du -sb "$1" 2>/dev/null | awk '{print $1}'; }
human() {
  local b="${1:-0}"; local u=(B K M G T P); local i=0
  while [ "$b" -ge 1024 ] && [ $i -lt 5 ]; do b=$(( b / 1024 )); i=$(( i + 1 )); done
  printf '%s%s' "$b" "${u[$i]}"
}

# ────────────────────────── store table printing + reference detection
declare -a SERVICE_PIDS=() MANUAL_SERVE_PIDS=()
detect_daemons() {
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl is-active ollama >/dev/null 2>&1; then
    SERVICE_PIDS+=("$(systemctl show -p MainPID --value ollama 2>/dev/null)")
  fi
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r pid; do
      # skip the systemd MainPID we already recorded
      local skip=0
      for sp in "${SERVICE_PIDS[@]:-}"; do [ "$pid" = "$sp" ] && skip=1; done
      [ $skip = 1 ] && continue
      MANUAL_SERVE_PIDS+=("$pid")
    done < <(pgrep -f 'ollama serve' 2>/dev/null || true)
  fi
}

print_store_table() {
  printf '%-4s %-45s %-8s %-22s %-9s %-9s %s\n' "  #" "Path" "Size" "Filesystem" "Manifests" "Blobs" "Owner"
  printf '%-4s %-45s %-8s %-22s %-9s %-9s %s\n' " ──" "────" "────" "──────────" "─────────" "─────" "─────"
  local i=0
  for s in "${STORE_PATHS[@]}"; do
    i=$((i + 1))
    local m b sz fs own
    m=$(count_manifests "$s"); b=$(count_blobs "$s")
    sz=$(human "$(size_of "$s")"); fs=$(fs_of "$s"); own=$(own_of "$s")
    printf '%-4s %-45s %-8s %-22s %-9s %-9s %s\n' "[$i]" "$s" "$sz" "$fs" "$m" "$b" "$own"
  done
}

# ─────────────────────────────────────── mount-point candidates for dest
print_dest_candidates() {
  hdr "Available mount points for unified storage:"
  df -h --output=target,size,avail,fstype 2>/dev/null \
    | awk 'NR==1 || ($1 ~ /^\/(home|srv|var|opt|mnt|media|data)/) || $1=="/"' \
    | awk 'NR==1 {print "  "$0; next} !seen[$1]++ {print "  "$0}'
}

# ────────────────────────────────────────── transfer-strategy selection
# Echoes one of: "mv" (same fs), "reflink" (CoW), "rsync" (cross-fs)
transfer_strategy() {
  local src="$1" dst="$2"
  local sd dd
  sd=$(dev_of "$src" 2>/dev/null || echo a)
  dd=$(dev_of "$dst" 2>/dev/null || echo b)
  if [ "$sd" = "$dd" ]; then
    echo "mv"; return
  fi
  # reflink test: try cp --reflink=always on a tiny file
  local probe="$dst/.reflink_probe_$$"
  if cp --reflink=always /dev/null "$probe" 2>/dev/null; then
    rm -f "$probe"
    echo "reflink"; return
  fi
  rm -f "$probe" 2>/dev/null || true
  echo "rsync"
}

# ──────────────────────────────────────── execute transfer per strategy
transfer_store() {
  local src="$1" dst="$2" strategy="$3" sudo_pfx="$4"
  case "$strategy" in
    mv)
      ok "same filesystem detected → using mv (instant)"
      # mv each subdir's contents so we merge into existing dst structure
      $sudo_pfx mkdir -p "$dst/manifests" "$dst/blobs"
      if [ -d "$src/manifests" ]; then
        find "$src/manifests" -mindepth 1 -maxdepth 1 -print0 \
          | xargs -0 -r -I{} $sudo_pfx mv -n "{}" "$dst/manifests/"
      fi
      if [ -d "$src/blobs" ]; then
        find "$src/blobs" -mindepth 1 -maxdepth 1 -print0 \
          | xargs -0 -r -I{} $sudo_pfx mv -n "{}" "$dst/blobs/"
      fi
      ;;
    reflink)
      ok "reflink-capable filesystem detected → using cp --reflink=auto"
      $sudo_pfx cp -a --reflink=auto -n "$src/." "$dst/"
      ;;
    rsync)
      ok "cross-filesystem copy → using rsync (NVMe-tuned)"
      # hardware-tuned: --whole-file (skip delta), --inplace, no compression,
      # ignore-existing for content-addressed dedup, progress2 for ETA
      $sudo_pfx rsync -aH --whole-file --inplace --no-compress \
        --ignore-existing --info=progress2 \
        "$src/" "$dst/"
      ;;
  esac
}

# ───────────────────────────────────────────────────────────── main flow
main() {
  banner
  discover_stores
  detect_daemons

  if [ ${#STORE_PATHS[@]} -eq 0 ]; then
    warn "No ollama model directories found. Nothing to unify."
    exit 0
  fi

  print_store_table

  hdr "Detected daemons:"
  if [ ${#SERVICE_PIDS[@]} -gt 0 ] && [ -n "${SERVICE_PIDS[0]:-}" ]; then
    say "  • ollama.service active (PID ${SERVICE_PIDS[0]})"
  else
    say "  • ollama.service: not active"
  fi
  if [ ${#MANUAL_SERVE_PIDS[@]} -gt 0 ]; then
    say "  • ${#MANUAL_SERVE_PIDS[@]} manual 'ollama serve' processes (PIDs: ${MANUAL_SERVE_PIDS[*]})"
  fi

  if [ ${#STORE_PATHS[@]} -eq 1 ]; then
    ok "Only one store found — your setup is already unified at: ${STORE_PATHS[0]}"
    exit 0
  fi

  print_dest_candidates

  # destination suggestion: largest fast mount that's already a store, else /srv/ollama/models
  local default_dst="${STORE_PATHS[0]}"
  local max_free=0
  for s in "${STORE_PATHS[@]}"; do
    local free
    free=$(df --output=avail "$s" 2>/dev/null | tail -1 | tr -d ' ')
    [ -z "$free" ] && continue
    if [ "$free" -gt "$max_free" ]; then max_free=$free; default_dst="$s"; fi
  done

  hdr "Destination"
  local DEST
  DEST=$(ask "Where to unify all models?" "$default_dst")
  DEST="$(readlink -m -- "$DEST")"

  # opt-ins
  hdr "Options"
  local DO_SYSTEMD=0 DO_SYMLINKS=0 DO_BASHRC=0 DO_SERVICE_USER=0
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl cat ollama >/dev/null 2>&1; then
    confirm "Update ollama.service via drop-in to use OLLAMA_MODELS=$DEST?" "Y" && DO_SYSTEMD=1 || true
    if [ "$DO_SYSTEMD" = 1 ]; then
      local cur_user
      cur_user=$(systemctl show -p User --value ollama 2>/dev/null)
      if [ -n "$cur_user" ] && [ "$cur_user" != "$USER" ]; then
        confirm "Service runs as '$cur_user'. Change to '$USER' for single-user simplification?" "Y" \
          && DO_SERVICE_USER=1 || true
      fi
    fi
  fi
  confirm "Replace each original store path with a symlink to $DEST (backward compat)?" "Y" \
    && DO_SYMLINKS=1 || true
  local SHELL_RC=""
  case "${SHELL##*/}" in
    bash) SHELL_RC="$HOME/.bashrc" ;;
    zsh)  SHELL_RC="$HOME/.zshrc" ;;
    fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
    *)    SHELL_RC="$HOME/.bashrc" ;;
  esac
  confirm "Add 'export OLLAMA_MODELS=$DEST' to $SHELL_RC?" "Y" && DO_BASHRC=1 || true

  # plan summary
  hdr "Plan"
  say "  Destination: $DEST"
  local total_bytes=0
  for s in "${STORE_PATHS[@]}"; do
    [ "$s" = "$DEST" ] && continue
    local strategy bytes
    strategy=$(transfer_strategy "$s" "$DEST" 2>/dev/null || echo rsync)
    bytes=$(size_of "$s")
    total_bytes=$((total_bytes + bytes))
    say "    $s  →  $DEST   ($(human "$bytes"), strategy: ${C_GRN}$strategy${C_RST})"
  done
  say "  Total to move: $(human "$total_bytes")"
  [ "$DO_SYSTEMD"      = 1 ] && say "  • Install systemd drop-in"
  [ "$DO_SERVICE_USER" = 1 ] && say "  • Change service User/Group to $USER"
  [ "$DO_SYMLINKS"     = 1 ] && say "  • Symlink originals → destination"
  [ "$DO_BASHRC"       = 1 ] && say "  • Add OLLAMA_MODELS export to $SHELL_RC"
  say "  • Stop running daemons, perform transfer, restart ollama.service"

  echo
  confirm "Proceed?" "N" || { warn "Aborted."; exit 1; }

  # ── sudo gate
  local SUDO=""
  if [ "$DO_SYSTEMD" = 1 ] || [ "$DO_SERVICE_USER" = 1 ] \
     || [[ "$DEST" =~ ^/(srv|var|opt|usr|etc) ]] \
     || [ ${#SERVICE_PIDS[@]} -gt 0 ] && [ -n "${SERVICE_PIDS[0]:-}" ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo required but not installed."; exit 2; }
    SUDO="sudo"
    say "${C_DIM}(sudo will prompt for your password)${C_RST}"
    $SUDO -v
  fi

  # ── stop daemons
  hdr "Stopping daemons"
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl is-active ollama >/dev/null 2>&1; then
    $SUDO systemctl stop ollama
    ok "ollama.service stopped"
  fi
  if [ ${#MANUAL_SERVE_PIDS[@]} -gt 0 ]; then
    for pid in "${MANUAL_SERVE_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${MANUAL_SERVE_PIDS[@]}"; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    ok "killed ${#MANUAL_SERVE_PIDS[@]} manual ollama serve(s)"
  fi
  # also nuke any remaining runners
  pkill -KILL -f 'ollama runner' 2>/dev/null || true
  sleep 1

  # ── ensure destination exists
  $SUDO mkdir -p "$DEST/manifests" "$DEST/blobs"

  # ── transfer each source
  hdr "Transferring"
  local started_at; started_at=$(date +%s)
  for s in "${STORE_PATHS[@]}"; do
    [ "$s" = "$DEST" ] && continue
    say "${C_BOLD}→ $s${C_RST}"
    local strategy
    strategy=$(transfer_strategy "$s" "$DEST")

    # special case: store has 0 manifests = orphan blobs, archive instead of merging
    local mcount; mcount=$(count_manifests "$s")
    if [ "$mcount" -eq 0 ] && [ "$s" != "$DEST" ]; then
      warn "  $s has 0 manifests (orphan blobs only) — archiving as .orphan-blobs"
      $SUDO mv "$s" "${s}.orphan-blobs"
      continue
    fi

    transfer_store "$s" "$DEST" "$strategy" "$SUDO"

    # post-transfer: rename original to .bak (or remove if empty after mv-merge)
    if [ -d "$s" ]; then
      if [ "$strategy" = "mv" ]; then
        # source dir likely empty now; just rmdir
        find "$s" -type d -empty -delete 2>/dev/null || true
        if [ -d "$s" ]; then
          $SUDO mv "$s" "${s}.bak"
        fi
      else
        $SUDO mv "$s" "${s}.bak"
      fi
      ok "  archived original: ${s}.bak"
    fi
  done
  local elapsed=$(( $(date +%s) - started_at ))
  local throughput=$(( total_bytes / (elapsed > 0 ? elapsed : 1) ))
  ok "transfer complete in ${elapsed}s  (~$(human "$throughput")/s)"

  # ── ownership + permissions on destination
  hdr "Finalizing destination"
  if [ "$DO_SERVICE_USER" = 1 ]; then
    $SUDO chown -R "$USER:$(id -gn)" "$DEST"
  fi
  $SUDO chmod -R u+rwX,g+rX,o+rX "$DEST"
  ok "permissions normalized"

  # ── systemd drop-in
  if [ "$DO_SYSTEMD" = 1 ]; then
    hdr "Updating systemd"
    $SUDO mkdir -p /etc/systemd/system/ollama.service.d
    {
      printf '[Service]\n'
      printf 'Environment=OLLAMA_MODELS=%s\n' "$DEST"
      if [ "$DO_SERVICE_USER" = 1 ]; then
        printf 'User=%s\n' "$USER"
        printf 'Group=%s\n' "$(id -gn)"
      fi
    } | $SUDO tee /etc/systemd/system/ollama.service.d/ollama-unify.conf >/dev/null
    $SUDO systemctl daemon-reload
    ok "drop-in installed: /etc/systemd/system/ollama.service.d/ollama-unify.conf"
  fi

  # ── symlinks for backward compat
  if [ "$DO_SYMLINKS" = 1 ]; then
    hdr "Creating backward-compat symlinks"
    for s in "${STORE_PATHS[@]}"; do
      [ "$s" = "$DEST" ] && continue
      [ -e "$s" ] && continue   # something still here (.bak rename failed?)
      $SUDO ln -s "$DEST" "$s"
      $SUDO chown -h "$USER:$(id -gn)" "$s" 2>/dev/null || true
      ok "  $s → $DEST"
    done
  fi

  # ── shell rc
  if [ "$DO_BASHRC" = 1 ]; then
    if ! grep -q "OLLAMA_MODELS=$DEST" "$SHELL_RC" 2>/dev/null; then
      {
        printf '\n# Unified Ollama model store (added by ollama-unify on %s)\n' "$(date +%F)"
        if [[ "$SHELL_RC" == *config.fish ]]; then
          printf 'set -gx OLLAMA_MODELS %s\n' "$DEST"
        else
          printf 'export OLLAMA_MODELS=%s\n' "$DEST"
        fi
      } >> "$SHELL_RC"
      ok "added export to $SHELL_RC (new shells only)"
    fi
  fi

  # ── restart service + verify
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl cat ollama >/dev/null 2>&1; then
    hdr "Restarting ollama.service"
    $SUDO systemctl start ollama
    sleep 3
    if systemctl is-active ollama >/dev/null 2>&1; then
      ok "ollama.service is active"
    else
      err "ollama.service failed to start — check 'journalctl -u ollama -n 50'"
    fi
  fi

  hdr "Verification"
  local mcount; mcount=$(count_manifests "$DEST")
  ok "$mcount manifests at $DEST"
  if [ "$HAS_CURL" = 1 ]; then
    local api_count
    api_count=$(curl -s --max-time 5 http://127.0.0.1:11434/api/tags 2>/dev/null \
      | grep -oP '"name"\s*:\s*"[^"]+"' | wc -l || true)
    [ -n "$api_count" ] && [ "$api_count" -gt 0 ] && ok "/api/tags reports $api_count models"
  fi

  hdr "Backups preserved (verify everything works, then reclaim):"
  for s in "${STORE_PATHS[@]}"; do
    [ "$s" = "$DEST" ] && continue
    [ -d "${s}.bak"           ] && say "  ${s}.bak  ($(du -sh "${s}.bak"           2>/dev/null | awk '{print $1}'))"
    [ -d "${s}.orphan-blobs"  ] && say "  ${s}.orphan-blobs  ($(du -sh "${s}.orphan-blobs"  2>/dev/null | awk '{print $1}'))"
  done

  hdr "Done."
  say "  Unified store:  $DEST"
  say "  Next steps:"
  say "    • open a new shell (or source $SHELL_RC) to pick up OLLAMA_MODELS"
  say "    • run: ${C_BOLD}ollama list${C_RST}    to confirm all models appear"
  say "    • after verifying, ${C_BOLD}rm -rf${C_RST} the .bak directories to reclaim disk"
}

main "$@"
