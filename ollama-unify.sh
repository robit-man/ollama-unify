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
# optionally rewires systemd + shell rc + backward-compat symlinks. It can also
# install dynamic GPU/host-memory guardrails that contain Ollama OOM failures.
#
# Safety: never deletes data. Renames originals to .bak / .orphan-blobs for you
# to remove after verifying.

set -euo pipefail

# ───────────────────────────────────────────────────────────────────── colors
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'; C_GRN=$'\033[32m'
  C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_RST=$'\033[0m'
else
  C_DIM=; C_BOLD=; C_RED=; C_GRN=; C_YEL=; C_CYN=; C_RST=
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
HAS_CURL=0; command -v curl >/dev/null 2>&1 && HAS_CURL=1
[ "$(uname -s)" = "Linux" ] || warn "Tested on Linux only. macOS/WSL may need manual tweaks."

# ─────────────────────────────────── dynamic Ollama resource safety profile
# Defaults can be tuned per run with OLLAMA_SAFE_* variables. GPU UUIDs are
# used instead of indices because their ordering can change between boots.
SAFETY_READY=0
SAFETY_NVIDIA_SMI=""
SAFETY_GPU_UUID_CSV=""
SAFETY_GPU_COUNT=0
SAFETY_VRAM_RESERVE_MIB=0
SAFETY_VRAM_RESERVE_BYTES=0
SAFETY_HOST_TOTAL_MIB=0
SAFETY_HOST_MEMORY_HIGH_MIB=0
SAFETY_HOST_MEMORY_MAX_MIB=0
SAFETY_CONTEXT_LENGTH=0
SAFETY_NUM_PARALLEL=0
SAFETY_MAX_LOADED_MODELS=0
SAFETY_MAX_QUEUE=0
SAFETY_KEEP_ALIVE=""
SAFETY_SWAP_MAX=""
declare -a SAFETY_GPU_UUIDS=() SAFETY_GPU_SUMMARIES=() SAFETY_EXCLUDED_GPU_SUMMARIES=()

trim_ws() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

require_uint_value() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || { err "$name must be an unsigned integer (got: $value)"; exit 2; }
}

build_safety_profile() {
  [ "$SAFETY_READY" = 1 ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || { warn "nvidia-smi not found; GPU guardrails unavailable."; return 1; }
  SAFETY_NVIDIA_SMI=$(command -v nvidia-smi)
  [[ "$SAFETY_NVIDIA_SMI" == /* ]] \
    || { warn "nvidia-smi did not resolve to an absolute path; GPU guardrails unavailable."; return 1; }
  [ -r /proc/meminfo ] || { warn "/proc/meminfo unavailable; host-memory guardrails unavailable."; return 1; }

  local inventory
  if ! inventory=$(nvidia-smi \
      --query-gpu=index,uuid,name,display_active,memory.total,compute_cap \
      --format=csv,noheader,nounits 2>/dev/null); then
    warn "nvidia-smi does not expose the telemetry needed for dynamic GPU selection."
    return 1
  fi

  local min_gpu_memory_mib="${OLLAMA_SAFE_MIN_GPU_MEMORY_MIB:-16384}"
  local min_compute_major="${OLLAMA_SAFE_MIN_COMPUTE_MAJOR:-7}"
  require_uint_value OLLAMA_SAFE_MIN_GPU_MEMORY_MIB "$min_gpu_memory_mib"
  require_uint_value OLLAMA_SAFE_MIN_COMPUTE_MAJOR "$min_compute_major"

  local min_selected_vram_mib=0
  local -a gpu_uuids=()
  SAFETY_GPU_SUMMARIES=()
  SAFETY_EXCLUDED_GPU_SUMMARIES=()

  while IFS=',' read -r raw_index raw_uuid raw_name raw_display raw_vram raw_compute; do
    local gpu_index gpu_uuid gpu_name display_active vram_mib compute_cap compute_major reason
    gpu_index=$(trim_ws "$raw_index")
    gpu_uuid=$(trim_ws "$raw_uuid")
    gpu_name=$(trim_ws "$raw_name")
    display_active=$(trim_ws "$raw_display")
    vram_mib=$(trim_ws "$raw_vram")
    compute_cap=$(trim_ws "$raw_compute")
    compute_major="${compute_cap%%.*}"
    reason=""

    if ! [[ "$vram_mib" =~ ^[0-9]+$ && "$compute_major" =~ ^[0-9]+$ ]]; then
      reason="incomplete CUDA telemetry"
    elif [[ "$gpu_name" == *"GT 1030"* ]]; then
      reason="desktop GT 1030"
    elif [ "$display_active" = "Enabled" ]; then
      reason="active display GPU"
    elif [ "$vram_mib" -lt "$min_gpu_memory_mib" ]; then
      reason="only ${vram_mib} MiB VRAM"
    elif [ "$compute_major" -lt "$min_compute_major" ]; then
      reason="CUDA compute capability ${compute_cap}"
    fi

    if [ -n "$reason" ]; then
      SAFETY_EXCLUDED_GPU_SUMMARIES+=("GPU $gpu_index: $gpu_name ($reason)")
      continue
    fi

    gpu_uuids+=("$gpu_uuid")
    SAFETY_GPU_SUMMARIES+=("GPU $gpu_index: $gpu_name, ${vram_mib} MiB, compute $compute_cap, $gpu_uuid")
    if [ "$min_selected_vram_mib" -eq 0 ] || [ "$vram_mib" -lt "$min_selected_vram_mib" ]; then
      min_selected_vram_mib="$vram_mib"
    fi
  done <<< "$inventory"

  [ ${#gpu_uuids[@]} -gt 0 ] || { warn "No eligible non-display CUDA inference GPUs detected; safety profile skipped."; return 1; }

  SAFETY_GPU_UUID_CSV=$(IFS=,; printf '%s' "${gpu_uuids[*]}")
  SAFETY_GPU_UUIDS=("${gpu_uuids[@]}")
  SAFETY_GPU_COUNT=${#gpu_uuids[@]}

  SAFETY_VRAM_RESERVE_MIB="${OLLAMA_SAFE_VRAM_RESERVE_MIB:-$((min_selected_vram_mib * 8 / 100))}"
  require_uint_value OLLAMA_SAFE_VRAM_RESERVE_MIB "$SAFETY_VRAM_RESERVE_MIB"
  [ "$SAFETY_VRAM_RESERVE_MIB" -lt 4096 ] && SAFETY_VRAM_RESERVE_MIB=4096
  [ "$SAFETY_VRAM_RESERVE_MIB" -gt 16384 ] && SAFETY_VRAM_RESERVE_MIB=16384
  SAFETY_VRAM_RESERVE_BYTES=$((SAFETY_VRAM_RESERVE_MIB * 1024 * 1024))

  SAFETY_HOST_TOTAL_MIB=$(awk '/^MemTotal:/ { print int($2 / 1024); exit }' /proc/meminfo)
  require_uint_value HOST_TOTAL_MIB "$SAFETY_HOST_TOTAL_MIB"
  local host_reserve_mib="${OLLAMA_SAFE_HOST_RESERVE_MIB:-$((SAFETY_HOST_TOTAL_MIB / 5))}"
  require_uint_value OLLAMA_SAFE_HOST_RESERVE_MIB "$host_reserve_mib"
  if [ "$SAFETY_HOST_TOTAL_MIB" -ge 131072 ] && [ "$host_reserve_mib" -lt 65536 ]; then
    host_reserve_mib=65536
  elif [ "$host_reserve_mib" -lt 8192 ]; then
    host_reserve_mib=8192
  fi

  SAFETY_HOST_MEMORY_MAX_MIB=$((SAFETY_HOST_TOTAL_MIB - host_reserve_mib))
  local throttle_band_mib=$((SAFETY_HOST_TOTAL_MIB / 20))
  [ "$throttle_band_mib" -lt 16384 ] && throttle_band_mib=16384
  SAFETY_HOST_MEMORY_HIGH_MIB=$((SAFETY_HOST_MEMORY_MAX_MIB - throttle_band_mib))
  if [ "$SAFETY_HOST_MEMORY_HIGH_MIB" -le 0 ] \
    || [ "$SAFETY_HOST_MEMORY_MAX_MIB" -le "$SAFETY_HOST_MEMORY_HIGH_MIB" ]; then
    err "dynamic host-memory guardrails leave too little RAM for Ollama"
    exit 2
  fi

  SAFETY_CONTEXT_LENGTH="${OLLAMA_SAFE_CONTEXT_LENGTH:-8192}"
  SAFETY_NUM_PARALLEL="${OLLAMA_SAFE_NUM_PARALLEL:-1}"
  SAFETY_MAX_QUEUE="${OLLAMA_SAFE_MAX_QUEUE:-64}"
  SAFETY_KEEP_ALIVE="${OLLAMA_SAFE_KEEP_ALIVE:-5m}"
  SAFETY_MAX_LOADED_MODELS="${OLLAMA_SAFE_MAX_LOADED_MODELS:-$((SAFETY_GPU_COUNT > 1 ? 2 : 1))}"
  SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-8G}"

  require_uint_value OLLAMA_SAFE_CONTEXT_LENGTH "$SAFETY_CONTEXT_LENGTH"
  require_uint_value OLLAMA_SAFE_NUM_PARALLEL "$SAFETY_NUM_PARALLEL"
  require_uint_value OLLAMA_SAFE_MAX_QUEUE "$SAFETY_MAX_QUEUE"
  require_uint_value OLLAMA_SAFE_MAX_LOADED_MODELS "$SAFETY_MAX_LOADED_MODELS"
  [ "$SAFETY_NUM_PARALLEL" -ge 1 ] || { err "OLLAMA_SAFE_NUM_PARALLEL must be at least 1"; exit 2; }
  [ "$SAFETY_MAX_LOADED_MODELS" -ge 1 ] || { err "OLLAMA_SAFE_MAX_LOADED_MODELS must be at least 1"; exit 2; }
  [[ "$SAFETY_KEEP_ALIVE" =~ ^[0-9]+(ms|s|m|h)$ ]] \
    || { err "OLLAMA_SAFE_KEEP_ALIVE must be a finite duration such as 5m"; exit 2; }
  [[ "$SAFETY_SWAP_MAX" =~ ^[0-9]+[KMGT]$ ]] \
    || { err "OLLAMA_SAFE_SWAP_MAX must be a systemd size such as 8G"; exit 2; }

  SAFETY_READY=1
}

print_safety_profile() {
  hdr "Dynamic Ollama safety profile"
  say "  Selected CUDA inference GPUs:"
  printf '    %s\n' "${SAFETY_GPU_SUMMARIES[@]}"
  if [ ${#SAFETY_EXCLUDED_GPU_SUMMARIES[@]} -gt 0 ]; then
    say "  Excluded GPUs:"
    printf '    %s\n' "${SAFETY_EXCLUDED_GPU_SUMMARIES[@]}"
  fi
  say "  VRAM reserve: ${SAFETY_VRAM_RESERVE_MIB} MiB per selected GPU"
  say "  Host RAM: throttle at ${SAFETY_HOST_MEMORY_HIGH_MIB} MiB; hard cap at ${SAFETY_HOST_MEMORY_MAX_MIB} MiB"
  say "  Scheduler: ${SAFETY_MAX_LOADED_MODELS} loaded model(s), ${SAFETY_NUM_PARALLEL} parallel request(s), ${SAFETY_CONTEXT_LENGTH}-token default context"
}

render_safety_service_directives() {
  printf '%s\n' \
    "Environment=\"CUDA_VISIBLE_DEVICES=${SAFETY_GPU_UUID_CSV}\"" \
    "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
    "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" \
    "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
    "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" \
    "Environment=\"OLLAMA_VULKAN=0\"" \
    "Environment=\"OLLAMA_IGPU_ENABLE=0\"" \
    "Environment=\"OLLAMA_MAX_LOADED_MODELS=${SAFETY_MAX_LOADED_MODELS}\"" \
    "Environment=\"OLLAMA_NUM_PARALLEL=${SAFETY_NUM_PARALLEL}\"" \
    "Environment=\"OLLAMA_CONTEXT_LENGTH=${SAFETY_CONTEXT_LENGTH}\"" \
    "Environment=\"OLLAMA_KEEP_ALIVE=${SAFETY_KEEP_ALIVE}\"" \
    "Environment=\"OLLAMA_MAX_QUEUE=${SAFETY_MAX_QUEUE}\"" \
    "Environment=\"OLLAMA_GPU_OVERHEAD=${SAFETY_VRAM_RESERVE_BYTES}\"" \
    "Environment=\"OLLAMA_FLASH_ATTENTION=1\"" \
    "Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\"" \
    "Environment=\"LLAMA_ARG_FIT=on\"" \
    "Environment=\"LLAMA_ARG_FIT_TARGET=${SAFETY_VRAM_RESERVE_MIB}\"" \
    "MemoryAccounting=yes" \
    "MemoryHigh=${SAFETY_HOST_MEMORY_HIGH_MIB}M" \
    "MemoryMax=${SAFETY_HOST_MEMORY_MAX_MIB}M" \
    "MemorySwapMax=${SAFETY_SWAP_MAX}" \
    "OOMPolicy=stop" \
    "Restart=on-failure" \
    "RestartSec=15s"
  local gpu_uuid
  for gpu_uuid in "${SAFETY_GPU_UUIDS[@]}"; do
    printf 'ExecStartPre=%s --id=%s --query-gpu=uuid,memory.total,compute_cap --format=csv,noheader,nounits\n' \
      "$SAFETY_NVIDIA_SMI" "$gpu_uuid"
  done
}

preview_safety_profile() {
  banner
  build_safety_profile || exit 1
  print_safety_profile
  hdr "Generated late-priority systemd drop-in"
  printf '%s\n' \
    "# Managed by ollama-unify" \
    "[Unit]" \
    "StartLimitIntervalSec=5min" \
    "StartLimitBurst=4" \
    "" \
    "[Service]"
  render_safety_service_directives
}

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
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  case "$strategy" in
    mv)
      ok "same filesystem detected → using mv (instant)"
      # mv each subdir's contents so we merge into existing dst structure
      "${elevate[@]}" mkdir -p "$dst/manifests" "$dst/blobs"
      if [ -d "$src/manifests" ]; then
        find "$src/manifests" -mindepth 1 -maxdepth 1 -print0 \
          | xargs -0 -r -I{} "${elevate[@]}" mv -n "{}" "$dst/manifests/"
      fi
      if [ -d "$src/blobs" ]; then
        find "$src/blobs" -mindepth 1 -maxdepth 1 -print0 \
          | xargs -0 -r -I{} "${elevate[@]}" mv -n "{}" "$dst/blobs/"
      fi
      ;;
    reflink)
      ok "reflink-capable filesystem detected → using cp --reflink=auto"
      "${elevate[@]}" cp -a --reflink=auto -n "$src/." "$dst/"
      ;;
    rsync)
      ok "cross-filesystem copy → using rsync (NVMe-tuned)"
      # hardware-tuned: --whole-file (skip delta), --inplace, no compression,
      # ignore-existing for content-addressed dedup, progress2 for ETA
      "${elevate[@]}" rsync -aH --whole-file --inplace --no-compress \
        --ignore-existing --info=progress2 \
        "$src/" "$dst/"
      ;;
  esac
}

# ───────────────────────────────────────────────────────────── main flow
main() {
  case "${1:-}" in
    --safety-preview)
      preview_safety_profile
      exit 0
      ;;
    -h|--help)
      say "Usage: ./ollama-unify.sh [--safety-preview]"
      say "  --safety-preview  Detect eligible GPUs and print OOM guardrails without changing the system."
      exit 0
      ;;
    "") ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac

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

  local ALREADY_UNIFIED=0
  if [ ${#STORE_PATHS[@]} -eq 1 ]; then
    ok "Only one store found — your setup is already unified at: ${STORE_PATHS[0]}"
    ALREADY_UNIFIED=1
  fi

  local DEST default_dst max_free
  if [ "$ALREADY_UNIFIED" = 1 ]; then
    DEST="${STORE_PATHS[0]}"
  else
    print_dest_candidates

    # destination suggestion: largest fast mount that's already a store, else /srv/ollama/models
    default_dst="${STORE_PATHS[0]}"
    max_free=0
    for s in "${STORE_PATHS[@]}"; do
      local free
      free=$(df --output=avail "$s" 2>/dev/null | tail -1 | tr -d ' ')
      [ -z "$free" ] && continue
      if [ "$free" -gt "$max_free" ]; then max_free=$free; default_dst="$s"; fi
    done

    hdr "Destination"
    DEST=$(ask "Where to unify all models?" "$default_dst")
    DEST="$(readlink -m -- "$DEST")"
  fi

  # opt-ins
  hdr "Options"
  local DO_SYSTEMD=0 DO_SYSTEMD_MODELS=0 DO_SAFETY=0 DO_SYMLINKS=0 DO_BASHRC=0 DO_SERVICE_USER=0
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl cat ollama >/dev/null 2>&1; then
    if confirm "Update ollama.service via drop-in to use OLLAMA_MODELS=$DEST?" "Y"; then
      DO_SYSTEMD=1
      DO_SYSTEMD_MODELS=1
    fi

    if build_safety_profile; then
      print_safety_profile
      if confirm "Apply these dynamic GPU and OOM safety guardrails to ollama.service?" "Y"; then
        DO_SYSTEMD=1
        DO_SAFETY=1
      fi
    fi

    if [ "$DO_SYSTEMD" = 1 ]; then
      local cur_user
      cur_user=$(systemctl show -p User --value ollama 2>/dev/null)
      if [ -n "$cur_user" ] && [ "$cur_user" != "$USER" ]; then
        if confirm "Service runs as '$cur_user'. Change to '$USER' for single-user simplification?" "Y"; then
          DO_SERVICE_USER=1
        fi
      fi
    fi
  fi
  if [ "$ALREADY_UNIFIED" = 0 ]; then
    if confirm "Replace each original store path with a symlink to $DEST (backward compat)?" "Y"; then
      DO_SYMLINKS=1
    fi
  fi
  local SHELL_RC=""
  case "${SHELL##*/}" in
    bash) SHELL_RC="$HOME/.bashrc" ;;
    zsh)  SHELL_RC="$HOME/.zshrc" ;;
    fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
    *)    SHELL_RC="$HOME/.bashrc" ;;
  esac
  if confirm "Add 'export OLLAMA_MODELS=$DEST' to $SHELL_RC?" "Y"; then
    DO_BASHRC=1
  fi

  if [ "$ALREADY_UNIFIED" = 1 ] && [ "$DO_SYSTEMD" = 0 ] && [ "$DO_BASHRC" = 0 ]; then
    ok "No changes selected."
    exit 0
  fi

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
  [ "$DO_SYSTEMD_MODELS" = 1 ] && say "  • Point ollama.service at the unified model store"
  [ "$DO_SAFETY"       = 1 ] && say "  • Install dynamic GPU/VRAM/host-memory OOM guardrails"
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

  if [ "$ALREADY_UNIFIED" = 0 ]; then
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
  fi

  # ── ownership + permissions on destination
  if [ "$ALREADY_UNIFIED" = 0 ] || [ "$DO_SERVICE_USER" = 1 ]; then
    hdr "Finalizing destination"
    if [ "$DO_SERVICE_USER" = 1 ]; then
      $SUDO chown -R "$USER:$(id -gn)" "$DEST"
    fi
    $SUDO chmod -R u+rwX,g+rX,o+rX "$DEST"
    ok "permissions normalized"
  fi

  # ── systemd drop-in
  if [ "$DO_SYSTEMD" = 1 ]; then
    hdr "Updating systemd"
    local dropin_dir="/etc/systemd/system/ollama.service.d"
    local dropin_file="$dropin_dir/zz-ollama-unify.conf"
    $SUDO mkdir -p "$dropin_dir"
    {
      printf '# Managed by ollama-unify — generated %s\n' "$(date -Is)"
      if [ "$DO_SAFETY" = 1 ]; then
        printf '[Unit]\n'
        printf 'StartLimitIntervalSec=5min\n'
        printf 'StartLimitBurst=4\n\n'
      fi
      printf '[Service]\n'
      if [ "$DO_SYSTEMD_MODELS" = 1 ]; then
        printf 'Environment="OLLAMA_MODELS=%s"\n' "$DEST"
      fi
      if [ "$DO_SERVICE_USER" = 1 ]; then
        printf 'User=%s\n' "$USER"
        printf 'Group=%s\n' "$(id -gn)"
      fi
      if [ "$DO_SAFETY" = 1 ]; then
        render_safety_service_directives
      fi
    } | $SUDO tee "$dropin_file" >/dev/null
    $SUDO systemctl daemon-reload
    if command -v systemd-analyze >/dev/null 2>&1; then
      $SUDO systemd-analyze verify ollama.service
    fi
    ok "late-priority drop-in installed: $dropin_file"
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
    # grep completes before this distinct append; it does not read while writing.
    # shellcheck disable=SC2094
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
