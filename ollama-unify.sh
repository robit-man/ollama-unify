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
require_migration_tools() {
  require rsync; require du; require df; require find; require stat; require awk; require sort
}

HAS_SUDO=0; command -v sudo >/dev/null 2>&1 && HAS_SUDO=1
HAS_SYSTEMD=0; SYSTEMD_VERSION=0
if command -v systemctl >/dev/null 2>&1 && systemctl --version >/dev/null 2>&1; then
  HAS_SYSTEMD=1
  SYSTEMD_VERSION=$(systemctl --version | awk 'NR==1 {print $2; exit}')
  [[ "$SYSTEMD_VERSION" =~ ^[0-9]+$ ]] || SYSTEMD_VERSION=0
fi
HAS_CURL=0; command -v curl >/dev/null 2>&1 && HAS_CURL=1

# ─────────────────────────────────── portable host + accelerator classifier
# The safety layer always produces a scheduler/host-memory profile. Accelerator
# discovery is capability-driven and degrades through CUDA → ROCm → Vulkan →
# Metal → CPU unless OLLAMA_SAFE_BACKEND explicitly selects an available one.
SAFETY_READY=0
HOST_OS=""
HOST_ARCH=""
HOST_NAME=""
HOST_CPU=""
HOST_CPU_CORES=0
HOST_VIRTUALIZATION="none"
HOST_SERVICE_MANAGER="none"
HOST_MEMORY_SOURCE=""
SAFETY_PHYSICAL_MEMORY_MIB=0
SAFETY_HOST_TOTAL_MIB=0
SAFETY_HOST_CLASS=""
SAFETY_BACKEND="cpu"
SAFETY_BACKEND_CLASS="fallback"
SAFETY_BACKEND_REASON=""
SAFETY_DEVICE_COUNT=0
SAFETY_SHARED_ACCELERATOR=0
SAFETY_MIN_DEVICE_MEMORY_MIB=0
SAFETY_AGGREGATE_DEVICE_MEMORY_MIB=0
SAFETY_DEVICE_MEMORY_KNOWN=0
SAFETY_DEDICATED_VRAM_RATIO_PERCENT=0
SAFETY_VRAM_RESERVE_MIB=0
SAFETY_VRAM_RESERVE_BYTES=0
SAFETY_HOST_RESERVE_MIB=0
SAFETY_HOST_MEMORY_HIGH_MIB=0
SAFETY_HOST_MEMORY_MAX_MIB=0
SAFETY_LARGEST_MODEL_MIB=0
SAFETY_LARGEST_MODEL_SOURCE=""
SAFETY_OBSERVED_HOST_MIB=0
SAFETY_HOST_LIMIT_SOURCE=""
SAFETY_GPU_STRICT=0
SAFETY_SCHED_SPREAD=0
SAFETY_CONTEXT_LENGTH=0
SAFETY_NUM_PARALLEL=0
SAFETY_MAX_LOADED_MODELS=0
SAFETY_MAX_QUEUE=0
SAFETY_KEEP_ALIVE=""
SAFETY_SWAP_MAX=""
SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT=20
SAFETY_CPU_QUOTA_PERCENT=400
SAFETY_CPU_WEIGHT=10
SAFETY_IO_WEIGHT=10
SAFETY_RESTART_POLICY="no"
SAFETY_PREFLIGHT_PATH="/usr/local/libexec/ollama-unify-memory-preflight"

CUDA_TOOL=""; CUDA_COUNT=0; CUDA_MIN_VRAM_MIB=0; CUDA_TOTAL_VRAM_MIB=0; CUDA_SHARED=0
ROCM_TOOL=""; ROCM_COUNT=0; ROCM_MIN_VRAM_MIB=0; ROCM_TOTAL_VRAM_MIB=0; ROCM_KNOWN_VRAM_COUNT=0; ROCM_SHARED=0
VULKAN_TOOL=""; VULKAN_COUNT=0; VULKAN_SHARED=0
METAL_COUNT=0
declare -a ACCELERATOR_SUMMARIES=() SAFETY_DEVICE_IDS=() SAFETY_SELECTED_SUMMARIES=()
declare -a SAFETY_PREFLIGHT_DIRECTIVES=()
declare -a CUDA_IDS=() CUDA_SUMMARIES=() CUDA_PREFLIGHT=()
declare -a ROCM_IDS=() ROCM_SUMMARIES=() ROCM_PREFLIGHT=()
declare -a VULKAN_IDS=() VULKAN_SUMMARIES=() VULKAN_PREFLIGHT=()
declare -a METAL_SUMMARIES=()

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

detect_host_profile() {
  HOST_OS=$(uname -s 2>/dev/null || printf 'Unknown')
  HOST_ARCH=$(uname -m 2>/dev/null || printf 'unknown')
  HOST_NAME=$(hostname 2>/dev/null || printf 'unknown')
  HOST_CPU_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
  [[ "$HOST_CPU_CORES" =~ ^[0-9]+$ ]] || HOST_CPU_CORES=1

  case "$HOST_OS" in
    Linux)
      if [ -r /etc/os-release ]; then
        HOST_NAME=$(awk -F= '/^PRETTY_NAME=/{v=substr($0,index($0,"=")+1); gsub(/^"|"$/,"",v); print v; exit}' /etc/os-release)
      fi
      HOST_CPU=$(awk -F: '/^(model name|Hardware)[[:space:]]*:/{v=$2; sub(/^[[:space:]]+/,"",v); print v; exit}' /proc/cpuinfo 2>/dev/null)
      [ -n "$HOST_CPU" ] || HOST_CPU="$HOST_ARCH CPU"
      if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
        HOST_VIRTUALIZATION="wsl"
      elif [ -e /.dockerenv ]; then
        HOST_VIRTUALIZATION="container"
      elif command -v systemd-detect-virt >/dev/null 2>&1; then
        local detected_virt=""
        if detected_virt=$(systemd-detect-virt 2>/dev/null); then
          HOST_VIRTUALIZATION="$detected_virt"
        else
          HOST_VIRTUALIZATION="none"
        fi
      fi
      if [ -d /run/systemd/system ] && [ "$HAS_SYSTEMD" = 1 ]; then HOST_SERVICE_MANAGER="systemd"; fi
      ;;
    Darwin)
      HOST_NAME="macOS $(sw_vers -productVersion 2>/dev/null || true)"
      HOST_CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || printf '%s CPU' "$HOST_ARCH")
      HOST_SERVICE_MANAGER="launchd"
      ;;
    FreeBSD)
      HOST_NAME="FreeBSD $(uname -r 2>/dev/null || true)"
      HOST_CPU=$(sysctl -n hw.model 2>/dev/null || printf '%s CPU' "$HOST_ARCH")
      HOST_SERVICE_MANAGER="rc.d"
      ;;
    *) HOST_CPU="$HOST_ARCH CPU" ;;
  esac

  local physical_mib=0
  if [ -r /proc/meminfo ]; then
    physical_mib=$(awk '/^MemTotal:/ { print int($2 / 1024); exit }' /proc/meminfo)
    HOST_MEMORY_SOURCE="/proc/meminfo"
  elif command -v sysctl >/dev/null 2>&1; then
    local memory_bytes
    memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || sysctl -n hw.physmem 2>/dev/null || printf '0')
    if [[ "$memory_bytes" =~ ^[0-9]+$ ]]; then physical_mib=$((memory_bytes / 1024 / 1024)); fi
    HOST_MEMORY_SOURCE="sysctl"
  fi
  if ! [[ "$physical_mib" =~ ^[0-9]+$ ]] || [ "$physical_mib" -lt 1024 ]; then
    physical_mib=4096
    HOST_MEMORY_SOURCE="conservative 4 GiB fallback"
  fi
  SAFETY_PHYSICAL_MEMORY_MIB="$physical_mib"
  SAFETY_HOST_TOTAL_MIB="$physical_mib"

  local limit_file="" limit_raw="" limit_mib=0
  if [ -r /sys/fs/cgroup/memory.max ]; then
    limit_file=/sys/fs/cgroup/memory.max
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    limit_file=/sys/fs/cgroup/memory/memory.limit_in_bytes
  fi
  if [ -n "$limit_file" ]; then
    limit_raw=$(tr -d '[:space:]' < "$limit_file")
    if [[ "$limit_raw" =~ ^[0-9]+$ ]]; then
      limit_mib=$(awk -v bytes="$limit_raw" 'BEGIN { printf "%.0f", bytes / 1048576 }')
      if [ "$limit_mib" -ge 1024 ] && [ "$limit_mib" -lt "$SAFETY_HOST_TOTAL_MIB" ]; then
        SAFETY_HOST_TOTAL_MIB="$limit_mib"
        HOST_MEMORY_SOURCE="$HOST_MEMORY_SOURCE, constrained by cgroup"
      fi
    fi
  fi
  if [ -n "${OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB "$OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB"
    [ "$OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB" -ge 1024 ] \
      || { err "OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB must be at least 1024"; exit 2; }
    SAFETY_HOST_TOTAL_MIB="$OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB"
    HOST_MEMORY_SOURCE="explicit override"
  fi

  case "$SAFETY_HOST_TOTAL_MIB" in
    ''|*[!0-9]*) SAFETY_HOST_CLASS="unknown" ;;
    *)
      if [ "$SAFETY_HOST_TOTAL_MIB" -lt 8192 ]; then SAFETY_HOST_CLASS="constrained"
      elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 32768 ]; then SAFETY_HOST_CLASS="personal"
      elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 131072 ]; then SAFETY_HOST_CLASS="workstation"
      else SAFETY_HOST_CLASS="memory-rich server"
      fi
      ;;
  esac
}

detect_cuda_devices() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  CUDA_TOOL=$(command -v nvidia-smi)
  [[ "$CUDA_TOOL" == /* ]] || { CUDA_TOOL=""; return 0; }
  local inventory=""
  inventory=$("$CUDA_TOOL" --query-gpu=index,uuid,name,display_active,memory.total,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null) || return 0

  local min_vram="${OLLAMA_SAFE_MIN_GPU_MEMORY_MIB:-4096}"
  local min_compute="${OLLAMA_SAFE_MIN_COMPUTE_MAJOR:-5}"
  require_uint_value OLLAMA_SAFE_MIN_GPU_MEMORY_MIB "$min_vram"
  require_uint_value OLLAMA_SAFE_MIN_COMPUTE_MAJOR "$min_compute"
  local -a dedicated_ids=() dedicated_summaries=() shared_ids=() shared_summaries=()
  local dedicated_min=0 dedicated_total=0 shared_min=0 shared_total=0

  while IFS=',' read -r raw_index raw_uuid raw_name raw_display raw_vram raw_compute; do
    local index uuid name display vram compute major role summary
    index=$(trim_ws "$raw_index"); uuid=$(trim_ws "$raw_uuid"); name=$(trim_ws "$raw_name")
    display=$(trim_ws "$raw_display"); vram=$(trim_ws "$raw_vram"); compute=$(trim_ws "$raw_compute")
    major="${compute%%.*}"
    if ! [[ "$vram" =~ ^[0-9]+$ && "$major" =~ ^[0-9]+$ && "$uuid" == GPU-* ]]; then
      ACCELERATOR_SUMMARIES+=("[cuda/unusable] GPU $index: $name — incomplete CUDA telemetry")
    elif [ "$major" -lt "$min_compute" ]; then
      ACCELERATOR_SUMMARIES+=("[cuda/legacy] GPU $index: $name, ${vram} MiB, compute $compute — below compute ${min_compute}.x")
    elif [ "$vram" -lt "$min_vram" ]; then
      ACCELERATOR_SUMMARIES+=("[cuda/constrained] GPU $index: $name, ${vram} MiB, compute $compute — below ${min_vram} MiB safety floor")
    else
      if [ "$display" = "Enabled" ]; then role="shared-display"; else role="dedicated"; fi
      summary="GPU $index: $name, ${vram} MiB, compute $compute, $uuid ($role)"
      ACCELERATOR_SUMMARIES+=("[cuda/$role] $summary")
      if [ "$role" = "dedicated" ]; then
        dedicated_ids+=("$uuid"); dedicated_summaries+=("$summary")
        dedicated_total=$((dedicated_total + vram))
        if [ "$dedicated_min" -eq 0 ] || [ "$vram" -lt "$dedicated_min" ]; then dedicated_min="$vram"; fi
      else
        shared_ids+=("$uuid"); shared_summaries+=("$summary")
        shared_total=$((shared_total + vram))
        if [ "$shared_min" -eq 0 ] || [ "$vram" -lt "$shared_min" ]; then shared_min="$vram"; fi
      fi
    fi
  done <<< "$inventory"

  if [ ${#dedicated_ids[@]} -gt 0 ]; then
    CUDA_IDS=("${dedicated_ids[@]}"); CUDA_SUMMARIES=("${dedicated_summaries[@]}")
    CUDA_MIN_VRAM_MIB="$dedicated_min"; CUDA_TOTAL_VRAM_MIB="$dedicated_total"; CUDA_SHARED=0
  elif [ ${#shared_ids[@]} -gt 0 ]; then
    CUDA_IDS=("${shared_ids[@]}"); CUDA_SUMMARIES=("${shared_summaries[@]}")
    CUDA_MIN_VRAM_MIB="$shared_min"; CUDA_TOTAL_VRAM_MIB="$shared_total"; CUDA_SHARED=1
  fi
  CUDA_COUNT=${#CUDA_IDS[@]}
  local uuid
  for uuid in "${CUDA_IDS[@]}"; do
    CUDA_PREFLIGHT+=("ExecStartPre=$CUDA_TOOL --id=$uuid --query-gpu=uuid,memory.total,compute_cap --format=csv,noheader,nounits")
  done
}

detect_rocm_devices() {
  local inventory="" tool=""
  if command -v amd-smi >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
    tool=$(command -v amd-smi)
    local amd_json=""
    amd_json=$("$tool" static --json 2>/dev/null) || amd_json=""
    if [ -n "$amd_json" ]; then
      inventory=$(printf '%s\n' "$amd_json" | jq -r '
        (if type == "array" then to_entries elif type == "object" then to_entries else [] end)[] |
        .key as $ordinal | .value as $g |
        [($g.gpu // $g.gpu_id // $g.id // $ordinal),
         ($g.uuid // $g.asic.uuid // $g.gpu_uuid // ""),
         ($g.asic.market_name // $g.asic.name // $g.board.product_name // $g.name // "AMD GPU"),
         ($g.vram.size.value // $g.vram.size // 0)] | @tsv' 2>/dev/null || true)
    fi
  fi
  if [ -z "$inventory" ] && command -v rocminfo >/dev/null 2>&1; then
    tool=$(command -v rocminfo)
    inventory=$("$tool" 2>/dev/null | awk '
      function flush() { if (is_gpu) { print ordinal "\t" uuid "\t" market "\t0"; ordinal++ } }
      /^[[:space:]]*Agent[[:space:]][0-9]+/ { flush(); is_gpu=0; uuid=""; market="AMD ROCm GPU"; next }
      /^[[:space:]]*Device Type:/ { if ($NF == "GPU") is_gpu=1; next }
      /^[[:space:]]*Uuid:/ { uuid=$NF; next }
      /^[[:space:]]*Marketing Name:/ { sub(/^[^:]*:[[:space:]]*/,""); market=$0; next }
      END { flush() }' || true)
  fi
  [ -n "$inventory" ] || return 0
  ROCM_TOOL="$tool"
  local ordinal=0
  while IFS=$'\t' read -r raw_id raw_uuid raw_name raw_vram; do
    local id uuid name vram role summary
    id=$(trim_ws "$raw_id"); uuid=$(trim_ws "$raw_uuid"); name=$(trim_ws "$raw_name"); vram=$(trim_ws "$raw_vram")
    [[ "$vram" =~ ^[0-9]+$ ]] || vram=0
    if [ -n "$uuid" ] && [ "$uuid" != "N/A" ]; then id="$uuid"; else id="$ordinal"; fi
    if [ "$vram" -eq 0 ]; then role="unknown-memory"; ROCM_SHARED=1
    elif [ "$vram" -lt 4096 ]; then role="shared-or-constrained"; ROCM_SHARED=1
    else role="discrete"; fi
    summary="GPU $ordinal: $name"
    [ "$vram" -gt 0 ] && summary="$summary, ${vram} MiB"
    summary="$summary, id $id ($role)"
    ROCM_IDS+=("$id"); ROCM_SUMMARIES+=("$summary")
    ACCELERATOR_SUMMARIES+=("[rocm/$role] $summary")
    if [ "$vram" -gt 0 ] && { [ "$ROCM_MIN_VRAM_MIB" -eq 0 ] || [ "$vram" -lt "$ROCM_MIN_VRAM_MIB" ]; }; then
      ROCM_MIN_VRAM_MIB="$vram"
    fi
    if [ "$vram" -gt 0 ]; then
      ROCM_TOTAL_VRAM_MIB=$((ROCM_TOTAL_VRAM_MIB + vram))
      ROCM_KNOWN_VRAM_COUNT=$((ROCM_KNOWN_VRAM_COUNT + 1))
    fi
    ordinal=$((ordinal + 1))
  done <<< "$inventory"
  ROCM_COUNT=${#ROCM_IDS[@]}
  if [ "$ROCM_COUNT" -gt 0 ]; then
    if [[ "$ROCM_TOOL" == */amd-smi ]]; then ROCM_PREFLIGHT+=("ExecStartPre=$ROCM_TOOL list")
    else ROCM_PREFLIGHT+=("ExecStartPre=$ROCM_TOOL"); fi
  fi
}

detect_vulkan_devices() {
  command -v vulkaninfo >/dev/null 2>&1 || return 0
  VULKAN_TOOL=$(command -v vulkaninfo)
  [[ "$VULKAN_TOOL" == /* ]] || { VULKAN_TOOL=""; return 0; }
  local inventory=""
  inventory=$("$VULKAN_TOOL" --summary 2>/dev/null | awk '
    function flush() { if (seen && name != "") print idx "\t" name "\t" dtype }
    /^[[:space:]]*GPU[0-9]+:/ { flush(); idx=$1; sub(/^GPU/,"",idx); sub(/:$/,"",idx); name=""; dtype="unknown"; seen=1; next }
    /^[[:space:]]*deviceName[[:space:]]*=/ { sub(/^[^=]*=[[:space:]]*/,""); name=$0; next }
    /^[[:space:]]*deviceType[[:space:]]*=/ { sub(/^[^=]*=[[:space:]]*/,""); dtype=$0; next }
    END { flush() }' || true)
  [ -n "$inventory" ] || { VULKAN_TOOL=""; return 0; }
  local -a discrete_ids=() discrete_summaries=() shared_ids=() shared_summaries=()
  while IFS=$'\t' read -r id name dtype; do
    local role summary
    case "$dtype" in
      *DISCRETE_GPU*) role="discrete" ;;
      *INTEGRATED_GPU*|*VIRTUAL_GPU*) role="shared" ;;
      *CPU*) ACCELERATOR_SUMMARIES+=("[vulkan/cpu-device] GPU $id: $name — not an accelerator"); continue ;;
      *) role="unknown" ;;
    esac
    summary="GPU $id: $name ($role, memory telemetry unavailable)"
    ACCELERATOR_SUMMARIES+=("[vulkan/$role] $summary")
    if [ "$role" = "discrete" ]; then discrete_ids+=("$id"); discrete_summaries+=("$summary")
    else shared_ids+=("$id"); shared_summaries+=("$summary"); fi
  done <<< "$inventory"
  if [ ${#discrete_ids[@]} -gt 0 ]; then
    VULKAN_IDS=("${discrete_ids[@]}"); VULKAN_SUMMARIES=("${discrete_summaries[@]}"); VULKAN_SHARED=0
  else
    VULKAN_IDS=("${shared_ids[@]}"); VULKAN_SUMMARIES=("${shared_summaries[@]}"); VULKAN_SHARED=1
  fi
  VULKAN_COUNT=${#VULKAN_IDS[@]}
  [ "$VULKAN_COUNT" -gt 0 ] && VULKAN_PREFLIGHT+=("ExecStartPre=$VULKAN_TOOL --summary")
}

detect_metal_devices() {
  [ "$HOST_OS" = "Darwin" ] || return 0
  if command -v system_profiler >/dev/null 2>&1; then
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      METAL_SUMMARIES+=("$name (Metal, unified/shared memory)")
      ACCELERATOR_SUMMARIES+=("[metal/unified] $name")
    done < <(system_profiler SPDisplaysDataType 2>/dev/null | awk -F: '/Chipset Model:/{sub(/^[[:space:]]+/,"",$2); print $2}')
  fi
  if [ ${#METAL_SUMMARIES[@]} -eq 0 ] && [ "$HOST_ARCH" = "arm64" ]; then
    METAL_SUMMARIES+=("Apple Silicon GPU (Metal, unified memory)")
    ACCELERATOR_SUMMARIES+=("[metal/unified] Apple Silicon GPU")
  fi
  METAL_COUNT=${#METAL_SUMMARIES[@]}
}

detect_unconfigured_accelerators() {
  [ ${#ACCELERATOR_SUMMARIES[@]} -eq 0 ] || return 0
  [ "$HOST_OS" = "Linux" ] || return 0
  if command -v lspci >/dev/null 2>&1; then
    while IFS= read -r device; do
      [ -n "$device" ] || continue
      ACCELERATOR_SUMMARIES+=("[pci/unconfigured] $device — no usable Ollama backend telemetry")
    done < <(lspci 2>/dev/null | awk 'tolower($0) ~ /vga compatible controller|3d controller|display controller/ {$1=""; sub(/^[[:space:]]+/,""); print}')
  fi
  if [ ${#ACCELERATOR_SUMMARIES[@]} -eq 0 ] && compgen -G '/dev/dri/renderD*' >/dev/null 2>&1; then
    ACCELERATOR_SUMMARIES+=("[drm/unconfigured] render nodes exist, but neither ROCm nor Vulkan telemetry is available")
  fi
}

select_safety_backend() {
  local requested="${OLLAMA_SAFE_BACKEND:-auto}"
  requested="${requested,,}"
  case "$requested" in auto|cuda|rocm|vulkan|metal|cpu) ;; *) err "OLLAMA_SAFE_BACKEND must be auto, cuda, rocm, vulkan, metal, or cpu"; exit 2 ;; esac
  if [ "$requested" = "auto" ]; then
    if [ "$HOST_OS" = "Darwin" ] && [ "$METAL_COUNT" -gt 0 ]; then requested="metal"
    elif [ "$CUDA_COUNT" -gt 0 ]; then requested="cuda"
    elif [ "$ROCM_COUNT" -gt 0 ]; then requested="rocm"
    elif [ "$VULKAN_COUNT" -gt 0 ]; then requested="vulkan"
    else requested="cpu"; fi
  fi

  case "$requested" in
    cuda)
      [ "$CUDA_COUNT" -gt 0 ] || { err "CUDA was requested but no eligible CUDA device was classified"; exit 2; }
      SAFETY_DEVICE_IDS=("${CUDA_IDS[@]}"); SAFETY_SELECTED_SUMMARIES=("${CUDA_SUMMARIES[@]}")
      SAFETY_PREFLIGHT_DIRECTIVES=("${CUDA_PREFLIGHT[@]}")
      SAFETY_DEVICE_COUNT="$CUDA_COUNT"
      SAFETY_AGGREGATE_DEVICE_MEMORY_MIB="$CUDA_TOTAL_VRAM_MIB"; SAFETY_DEVICE_MEMORY_KNOWN=1
      SAFETY_MIN_DEVICE_MEMORY_MIB="$CUDA_MIN_VRAM_MIB"; SAFETY_SHARED_ACCELERATOR="$CUDA_SHARED"
      SAFETY_BACKEND_CLASS=$([ "$CUDA_SHARED" = 1 ] && printf 'shared-display' || printf 'dedicated')
      SAFETY_BACKEND_REASON="highest-confidence native NVIDIA backend"
      ;;
    rocm)
      [ "$ROCM_COUNT" -gt 0 ] || { err "ROCm was requested but no ROCm device was classified"; exit 2; }
      SAFETY_DEVICE_IDS=("${ROCM_IDS[@]}"); SAFETY_SELECTED_SUMMARIES=("${ROCM_SUMMARIES[@]}")
      SAFETY_PREFLIGHT_DIRECTIVES=("${ROCM_PREFLIGHT[@]}")
      SAFETY_DEVICE_COUNT="$ROCM_COUNT"
      if [ "$ROCM_KNOWN_VRAM_COUNT" -eq "$ROCM_COUNT" ]; then
        SAFETY_AGGREGATE_DEVICE_MEMORY_MIB="$ROCM_TOTAL_VRAM_MIB"; SAFETY_DEVICE_MEMORY_KNOWN=1
      fi
      SAFETY_MIN_DEVICE_MEMORY_MIB="$ROCM_MIN_VRAM_MIB"; SAFETY_SHARED_ACCELERATOR="$ROCM_SHARED"
      SAFETY_BACKEND_CLASS=$([ "$ROCM_SHARED" = 1 ] && printf 'shared/integrated' || printf 'discrete')
      SAFETY_BACKEND_REASON="native AMD ROCm backend"
      ;;
    vulkan)
      [ "$VULKAN_COUNT" -gt 0 ] || { err "Vulkan was requested but no Vulkan accelerator was classified"; exit 2; }
      SAFETY_DEVICE_IDS=("${VULKAN_IDS[@]}"); SAFETY_SELECTED_SUMMARIES=("${VULKAN_SUMMARIES[@]}")
      SAFETY_PREFLIGHT_DIRECTIVES=("${VULKAN_PREFLIGHT[@]}")
      SAFETY_DEVICE_COUNT="$VULKAN_COUNT"
      SAFETY_SHARED_ACCELERATOR="$VULKAN_SHARED"; SAFETY_MIN_DEVICE_MEMORY_MIB=0
      SAFETY_BACKEND_CLASS=$([ "$VULKAN_SHARED" = 1 ] && printf 'shared/integrated' || printf 'discrete')
      SAFETY_BACKEND_REASON="portable GPU fallback with approximate memory telemetry"
      ;;
    metal)
      [ "$METAL_COUNT" -gt 0 ] || { err "Metal was requested but no Metal device was classified"; exit 2; }
      SAFETY_SELECTED_SUMMARIES=("${METAL_SUMMARIES[@]}")
      SAFETY_DEVICE_COUNT="$METAL_COUNT"
      SAFETY_SHARED_ACCELERATOR=1; SAFETY_MIN_DEVICE_MEMORY_MIB=0
      SAFETY_BACKEND_CLASS="unified-memory"; SAFETY_BACKEND_REASON="native Apple Metal backend"
      ;;
    cpu)
      SAFETY_DEVICE_COUNT=0; SAFETY_SHARED_ACCELERATOR=1; SAFETY_MIN_DEVICE_MEMORY_MIB=0
      SAFETY_SELECTED_SUMMARIES=("$HOST_CPU (${HOST_CPU_CORES} logical cores)")
      SAFETY_BACKEND_CLASS="host-memory"; SAFETY_BACKEND_REASON="no eligible accelerator or explicit CPU selection"
      ;;
  esac
  SAFETY_BACKEND="$requested"
}

detect_model_memory_profile() {
  SAFETY_LARGEST_MODEL_MIB=0
  SAFETY_LARGEST_MODEL_SOURCE=""
  SAFETY_OBSERVED_HOST_MIB=0

  if [ -n "${OLLAMA_SAFE_LARGEST_MODEL_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_LARGEST_MODEL_MIB "$OLLAMA_SAFE_LARGEST_MODEL_MIB"
    SAFETY_LARGEST_MODEL_MIB="$OLLAMA_SAFE_LARGEST_MODEL_MIB"
    SAFETY_LARGEST_MODEL_SOURCE="explicit override"
  else
    local -a roots=() candidates=()
    local candidate existing duplicate
    if [ -n "${OLLAMA_SAFE_MODEL_STORE:-}" ]; then
      candidates+=("$OLLAMA_SAFE_MODEL_STORE")
    else
      [ -n "${OLLAMA_MODELS:-}" ] && candidates+=("$OLLAMA_MODELS")
      candidates+=("/srv/ollama/models" "${HOME}/.ollama/models" \
        "/usr/share/ollama/.ollama/models" "/var/lib/ollama/.ollama/models" "/root/.ollama/models")
      candidates+=("${STORE_PATHS[@]:-}")

      local configured=""
      if [ "$HAS_SYSTEMD" = 1 ]; then
        configured=$(systemctl show ollama.service -p Environment --value 2>/dev/null \
          | grep -o 'OLLAMA_MODELS=[^[:space:]]*' | tail -n 1 || true)
        [ -n "$configured" ] && candidates+=("${configured#OLLAMA_MODELS=}")
      fi
      for existing in /etc/default/ollama /etc/environment; do
        [ -r "$existing" ] || continue
        while IFS= read -r configured; do
          configured="${configured#OLLAMA_MODELS=}"
          configured="${configured%\"}"; configured="${configured#\"}"
          configured="${configured%\'}"; configured="${configured#\'}"
          [ -n "$configured" ] && candidates+=("$configured")
        done < <(grep -E '^[[:space:]]*(export[[:space:]]+)?OLLAMA_MODELS=' "$existing" 2>/dev/null \
          | sed -E 's/^[[:space:]]*(export[[:space:]]+)?OLLAMA_MODELS=//' || true)
      done
    fi

    for candidate in "${candidates[@]}"; do
      [ -d "$candidate/manifests" ] || continue
      duplicate=0
      for existing in "${roots[@]:-}"; do
        [ "$candidate" = "$existing" ] && { duplicate=1; break; }
      done
      [ "$duplicate" = 1 ] || roots+=("$candidate")
    done

    local manifest bytes mib max_bytes=0
    for candidate in "${roots[@]:-}"; do
      while IFS= read -r -d '' manifest; do
        # Sum only inference payloads in each manifest. This includes split
        # model/projector/adapter/tensor layers while excluding templates,
        # licenses, prompts, and other metadata.
        bytes=$(awk 'BEGIN { RS="}" }
          /"mediaType"[[:space:]]*:[[:space:]]*"application\/vnd\.ollama\.image\.(model|projector|adapter|tensor)"/ {
            record=$0
            sub(/^.*"size"[[:space:]]*:[[:space:]]*/, "", record)
            sub(/[^0-9].*$/, "", record)
            if (record ~ /^[0-9]+$/) sum += record
          }
          END { printf "%.0f", sum + 0 }' "$manifest" 2>/dev/null || printf '0')
        [[ "$bytes" =~ ^[0-9]+$ ]] || bytes=0
        if [ "$bytes" -gt "$max_bytes" ]; then
          max_bytes="$bytes"
          SAFETY_LARGEST_MODEL_SOURCE="$manifest"
        fi
      done < <(find "$candidate/manifests" -type f -print0 2>/dev/null)
    done
    if [ "$max_bytes" -gt 0 ]; then
      mib=$(((max_bytes + 1048575) / 1048576))
      SAFETY_LARGEST_MODEL_MIB="$mib"
    fi
  fi

  if [ -n "${OLLAMA_SAFE_OBSERVED_HOST_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_OBSERVED_HOST_MIB "$OLLAMA_SAFE_OBSERVED_HOST_MIB"
    SAFETY_OBSERVED_HOST_MIB="$OLLAMA_SAFE_OBSERVED_HOST_MIB"
  elif command -v journalctl >/dev/null 2>&1; then
    local observed=""
    observed=$(journalctl -u ollama.service --no-pager --since '-30 days' --grep 'host memory' -n 512 2>/dev/null \
      | awk '/projected to use [0-9]+ MiB of host memory/ {
          line=$0
          sub(/^.*projected to use /, "", line)
          sub(/ .*/, "", line)
          if (line + 0 > max) max=line + 0
        }
        END { print max + 0 }' || true)
    [[ "$observed" =~ ^[0-9]+$ ]] && SAFETY_OBSERVED_HOST_MIB="$observed"
  fi
}

build_resource_limits() {
  # Host limits are measurements, not percentages selected for a particular
  # machine: normal pressure begins at the largest host projection Ollama has
  # reported, and the hard boundary is the largest installed inference payload.
  SAFETY_DEDICATED_VRAM_RATIO_PERCENT=0
  if [ "$SAFETY_DEVICE_MEMORY_KNOWN" = 1 ] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ] \
    && [ "$SAFETY_AGGREGATE_DEVICE_MEMORY_MIB" -gt 0 ]; then
    SAFETY_DEDICATED_VRAM_RATIO_PERCENT=$((SAFETY_AGGREGATE_DEVICE_MEMORY_MIB * 100 / SAFETY_HOST_TOTAL_MIB))
  fi

  local host_max=0 host_high=0
  if [ -n "${OLLAMA_SAFE_HOST_RESERVE_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_HOST_RESERVE_MIB "$OLLAMA_SAFE_HOST_RESERVE_MIB"
    [ "$OLLAMA_SAFE_HOST_RESERVE_MIB" -lt "$SAFETY_HOST_TOTAL_MIB" ] \
      || { err "OLLAMA_SAFE_HOST_RESERVE_MIB must be smaller than effective host memory"; exit 2; }
    host_max=$((SAFETY_HOST_TOTAL_MIB - OLLAMA_SAFE_HOST_RESERVE_MIB))
    host_high="$host_max"
    SAFETY_HOST_LIMIT_SOURCE="explicit host reserve"
  else
    [ "$SAFETY_LARGEST_MODEL_MIB" -gt 0 ] || {
      err "no installed Ollama inference payload was found; refusing to invent a host-memory limit"
      err "install a model, set OLLAMA_SAFE_MODEL_STORE, or explicitly set OLLAMA_SAFE_HOST_RESERVE_MIB"
      exit 2
    }
    host_max="$SAFETY_LARGEST_MODEL_MIB"
    [ "$SAFETY_OBSERVED_HOST_MIB" -gt "$host_max" ] && host_max="$SAFETY_OBSERVED_HOST_MIB"
    host_high="$SAFETY_OBSERVED_HOST_MIB"
    [ "$host_high" -gt 0 ] || host_high="$host_max"
    SAFETY_HOST_LIMIT_SOURCE="installed manifests and Ollama journal projections"
  fi
  if [ -n "${OLLAMA_SAFE_HOST_MEMORY_MAX_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_HOST_MEMORY_MAX_MIB "$OLLAMA_SAFE_HOST_MEMORY_MAX_MIB"
    host_max="$OLLAMA_SAFE_HOST_MEMORY_MAX_MIB"
    SAFETY_HOST_LIMIT_SOURCE="explicit host-memory boundary"
  fi
  if [ -n "${OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB "$OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB"
    host_high="$OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB"
  fi
  if [ "$host_max" -le 0 ] || [ "$host_max" -ge "$SAFETY_HOST_TOTAL_MIB" ]; then
    err "derived host-memory hard cap (${host_max} MiB) must be between 1 MiB and effective host RAM"
    exit 2
  fi
  if [ "$host_high" -le 0 ] || [ "$host_high" -gt "$host_max" ]; then
    err "host-memory throttle (${host_high} MiB) must be between 1 MiB and the hard cap"
    exit 2
  fi
  SAFETY_HOST_MEMORY_MAX_MIB="$host_max"
  SAFETY_HOST_MEMORY_HIGH_MIB="$host_high"
  SAFETY_HOST_RESERVE_MIB=$((SAFETY_HOST_TOTAL_MIB - SAFETY_HOST_MEMORY_MAX_MIB))

  # Ollama already measures live free VRAM when it schedules a load. Do not
  # subtract a guessed percentage a second time. A fixed carve-out exists only
  # when the operator explicitly requests one.
  SAFETY_VRAM_RESERVE_MIB=0
  if [ -n "${OLLAMA_SAFE_VRAM_RESERVE_MIB:-}" ] && [ "$SAFETY_BACKEND" != "cpu" ] && [ "$SAFETY_BACKEND" != "metal" ]; then
    require_uint_value OLLAMA_SAFE_VRAM_RESERVE_MIB "$OLLAMA_SAFE_VRAM_RESERVE_MIB"
    SAFETY_VRAM_RESERVE_MIB="$OLLAMA_SAFE_VRAM_RESERVE_MIB"
  fi
  if [ "$SAFETY_MIN_DEVICE_MEMORY_MIB" -gt 0 ] \
    && [ "$SAFETY_VRAM_RESERVE_MIB" -gt $((SAFETY_MIN_DEVICE_MEMORY_MIB / 2)) ]; then
    err "OLLAMA_SAFE_VRAM_RESERVE_MIB cannot exceed half of the smallest selected device"
    exit 2
  fi
  SAFETY_VRAM_RESERVE_BYTES=$((SAFETY_VRAM_RESERVE_MIB * 1024 * 1024))

  SAFETY_GPU_STRICT=0
  SAFETY_SCHED_SPREAD=0
  if [[ "$SAFETY_BACKEND" =~ ^(cuda|rocm)$ ]] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ]; then
    SAFETY_GPU_STRICT=1
    [ "$SAFETY_DEVICE_COUNT" -gt 1 ] && SAFETY_SCHED_SPREAD=1
  fi

  local default_context=8192 default_queue=64 default_models=1
  if [ "$SAFETY_HOST_TOTAL_MIB" -lt 8192 ]; then default_context=2048; default_queue=8
  elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 16384 ]; then default_context=4096; default_queue=16; fi
  if [ "$SAFETY_BACKEND" = "cpu" ] && [ "$SAFETY_HOST_TOTAL_MIB" -lt 32768 ] && [ "$default_context" -gt 4096 ]; then
    default_context=4096
  fi

  SAFETY_CONTEXT_LENGTH="${OLLAMA_SAFE_CONTEXT_LENGTH:-$default_context}"
  SAFETY_NUM_PARALLEL="${OLLAMA_SAFE_NUM_PARALLEL:-1}"
  SAFETY_MAX_QUEUE="${OLLAMA_SAFE_MAX_QUEUE:-$default_queue}"
  SAFETY_KEEP_ALIVE="${OLLAMA_SAFE_KEEP_ALIVE:-5m}"
  SAFETY_MAX_LOADED_MODELS="${OLLAMA_SAFE_MAX_LOADED_MODELS:-$default_models}"
  if [ "$SAFETY_GPU_STRICT" = 1 ]; then SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-0}"
  elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 16384 ]; then SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-2G}"
  else SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-8G}"; fi
  SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT="${OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT:-20}"
  SAFETY_CPU_QUOTA_PERCENT="${OLLAMA_SAFE_CPU_QUOTA_PERCENT:-400}"
  SAFETY_CPU_WEIGHT="${OLLAMA_SAFE_CPU_WEIGHT:-10}"
  SAFETY_IO_WEIGHT="${OLLAMA_SAFE_IO_WEIGHT:-10}"
  SAFETY_RESTART_POLICY="${OLLAMA_SAFE_RESTART_POLICY:-no}"

  local host_cpu_capacity=$((HOST_CPU_CORES * 100))
  [ "$host_cpu_capacity" -ge 100 ] || host_cpu_capacity=100

  require_uint_value OLLAMA_SAFE_CONTEXT_LENGTH "$SAFETY_CONTEXT_LENGTH"
  require_uint_value OLLAMA_SAFE_NUM_PARALLEL "$SAFETY_NUM_PARALLEL"
  require_uint_value OLLAMA_SAFE_MAX_QUEUE "$SAFETY_MAX_QUEUE"
  require_uint_value OLLAMA_SAFE_MAX_LOADED_MODELS "$SAFETY_MAX_LOADED_MODELS"
  require_uint_value OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT "$SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT"
  require_uint_value OLLAMA_SAFE_CPU_QUOTA_PERCENT "$SAFETY_CPU_QUOTA_PERCENT"
  require_uint_value OLLAMA_SAFE_CPU_WEIGHT "$SAFETY_CPU_WEIGHT"
  require_uint_value OLLAMA_SAFE_IO_WEIGHT "$SAFETY_IO_WEIGHT"
  [ "$SAFETY_NUM_PARALLEL" -ge 1 ] || { err "OLLAMA_SAFE_NUM_PARALLEL must be at least 1"; exit 2; }
  [ "$SAFETY_MAX_LOADED_MODELS" -ge 1 ] || { err "OLLAMA_SAFE_MAX_LOADED_MODELS must be at least 1"; exit 2; }
  if [ "$SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT" -lt 1 ] \
    || [ "$SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT" -gt 100 ]; then
    err "OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT must be between 1 and 100"; exit 2
  fi
  [ "$SAFETY_CPU_QUOTA_PERCENT" -ge 100 ] \
    || { err "OLLAMA_SAFE_CPU_QUOTA_PERCENT must be at least 100"; exit 2; }
  [ "$SAFETY_CPU_QUOTA_PERCENT" -gt "$host_cpu_capacity" ] && SAFETY_CPU_QUOTA_PERCENT="$host_cpu_capacity"
  if [ "$SAFETY_CPU_WEIGHT" -lt 1 ] || [ "$SAFETY_CPU_WEIGHT" -gt 10000 ]; then
    err "OLLAMA_SAFE_CPU_WEIGHT must be between 1 and 10000"; exit 2
  fi
  if [ "$SAFETY_IO_WEIGHT" -lt 1 ] || [ "$SAFETY_IO_WEIGHT" -gt 10000 ]; then
    err "OLLAMA_SAFE_IO_WEIGHT must be between 1 and 10000"; exit 2
  fi
  case "$SAFETY_RESTART_POLICY" in
    no|on-failure) ;;
    *) err "OLLAMA_SAFE_RESTART_POLICY must be no or on-failure"; exit 2 ;;
  esac
  [[ "$SAFETY_KEEP_ALIVE" =~ ^[0-9]+(ms|s|m|h)$ ]] || { err "OLLAMA_SAFE_KEEP_ALIVE must be a finite duration such as 5m"; exit 2; }
  [[ "$SAFETY_SWAP_MAX" =~ ^(0|[0-9]+[KMGT])$ ]] || { err "OLLAMA_SAFE_SWAP_MAX must be 0 or a systemd size such as 8G"; exit 2; }
}

build_safety_profile() {
  [ "$SAFETY_READY" = 1 ] && return 0
  detect_host_profile
  detect_cuda_devices
  detect_rocm_devices
  detect_vulkan_devices
  detect_metal_devices
  detect_unconfigured_accelerators
  select_safety_backend
  detect_model_memory_profile
  build_resource_limits
  SAFETY_READY=1
}

print_safety_profile() {
  hdr "Host classification"
  say "  Platform: $HOST_NAME ($HOST_OS/$HOST_ARCH; $HOST_VIRTUALIZATION)"
  say "  CPU: $HOST_CPU — $HOST_CPU_CORES logical cores"
  if [ "$SAFETY_PHYSICAL_MEMORY_MIB" -ne "$SAFETY_HOST_TOTAL_MIB" ]; then
    say "  Memory: ${SAFETY_PHYSICAL_MEMORY_MIB} MiB physical; ${SAFETY_HOST_TOTAL_MIB} MiB effective ($HOST_MEMORY_SOURCE)"
  else
    say "  Memory: ${SAFETY_HOST_TOTAL_MIB} MiB ($HOST_MEMORY_SOURCE)"
  fi
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    say "  Class: $SAFETY_HOST_CLASS; service manager: systemd $SYSTEMD_VERSION"
  else
    say "  Class: $SAFETY_HOST_CLASS; service manager: $HOST_SERVICE_MANAGER"
  fi

  hdr "Accelerator classification"
  if [ ${#ACCELERATOR_SUMMARIES[@]} -gt 0 ]; then printf '  %s\n' "${ACCELERATOR_SUMMARIES[@]}"
  else say "  No usable accelerator telemetry; CPU fallback is available."; fi

  hdr "Selected Ollama safety policy"
  say "  Backend: $SAFETY_BACKEND ($SAFETY_BACKEND_CLASS) — $SAFETY_BACKEND_REASON"
  printf '  Device: %s\n' "${SAFETY_SELECTED_SUMMARIES[@]}"
  if [ "$SAFETY_VRAM_RESERVE_MIB" -gt 0 ]; then
    say "  Device memory: explicit ${SAFETY_VRAM_RESERVE_MIB} MiB carve-out per selected accelerator"
  elif [ "$SAFETY_DEVICE_COUNT" -gt 0 ]; then
    say "  Device memory: live free-VRAM telemetry; no guessed fixed carve-out"
  fi
  if [ "$SAFETY_DEVICE_MEMORY_KNOWN" = 1 ] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ]; then
    say "  Aggregate dedicated device memory: ${SAFETY_AGGREGATE_DEVICE_MEMORY_MIB} MiB across ${SAFETY_DEVICE_COUNT} accelerator(s); ${SAFETY_DEDICATED_VRAM_RATIO_PERCENT}% of host RAM"
  fi
  if [ "$SAFETY_LARGEST_MODEL_MIB" -gt 0 ]; then
    say "  Model scan: largest installed inference payload ${SAFETY_LARGEST_MODEL_MIB} MiB ($SAFETY_LARGEST_MODEL_SOURCE)"
  fi
  if [ "$SAFETY_OBSERVED_HOST_MIB" -gt 0 ]; then
    say "  Ollama history: largest observed host projection ${SAFETY_OBSERVED_HOST_MIB} MiB"
  fi
  say "  Host memory: throttle at ${SAFETY_HOST_MEMORY_HIGH_MIB} MiB; hard cap at ${SAFETY_HOST_MEMORY_MAX_MIB} MiB; ${SAFETY_HOST_RESERVE_MIB} MiB remains outside the cgroup"
  say "  Host limit basis: $SAFETY_HOST_LIMIT_SOURCE"
  say "  Scheduler: ${SAFETY_MAX_LOADED_MODELS} model(s), ${SAFETY_NUM_PARALLEL} parallel request(s), ${SAFETY_CONTEXT_LENGTH}-token context, queue ${SAFETY_MAX_QUEUE}"
  if [ "$SAFETY_GPU_STRICT" = 1 ]; then
    say "  GPU policy: all layers on dedicated accelerators; spread=$SAFETY_SCHED_SPREAD; unified spill and pinned-host buffers disabled"
  fi
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    say "  Containment: memory PSI ${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}%; CPU quota ${SAFETY_CPU_QUOTA_PERCENT}%; restart $SAFETY_RESTART_POLICY"
  fi
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ] && [ "$SYSTEMD_VERSION" -lt 231 ]; then
    warn "systemd $SYSTEMD_VERSION is too old for MemoryHigh/MemoryMax; scheduler limits still apply."
  elif [ "$HOST_SERVICE_MANAGER" != "systemd" ]; then
    warn "Native cgroup OOM containment is unavailable under $HOST_SERVICE_MANAGER; scheduler limits still apply."
  fi
}

csv_from_array() { local IFS=,; printf '%s' "$*"; }

render_safety_environment_directives() {
  local ids
  ids=$(csv_from_array "${SAFETY_DEVICE_IDS[@]}")
  case "$SAFETY_BACKEND" in
    cuda)
      printf '%s\n' "Environment=\"CUDA_VISIBLE_DEVICES=$ids\"" "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
        "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" "Environment=\"OLLAMA_VULKAN=0\"" "Environment=\"OLLAMA_IGPU_ENABLE=0\""
      ;;
    rocm)
      printf '%s\n' "Environment=\"ROCR_VISIBLE_DEVICES=$ids\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" "Environment=\"OLLAMA_VULKAN=0\"" \
        "Environment=\"OLLAMA_IGPU_ENABLE=$SAFETY_SHARED_ACCELERATOR\""
      ;;
    vulkan)
      printf '%s\n' "Environment=\"CUDA_VISIBLE_DEVICES=-1\"" "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
        "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=$ids\"" "Environment=\"OLLAMA_VULKAN=1\"" \
        "Environment=\"OLLAMA_IGPU_ENABLE=$SAFETY_SHARED_ACCELERATOR\""
      ;;
    cpu)
      printf '%s\n' "Environment=\"CUDA_VISIBLE_DEVICES=-1\"" "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
        "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" "Environment=\"OLLAMA_VULKAN=0\"" "Environment=\"OLLAMA_IGPU_ENABLE=0\""
      ;;
  esac
  printf '%s\n' \
    "Environment=\"OLLAMA_MAX_LOADED_MODELS=${SAFETY_MAX_LOADED_MODELS}\"" \
    "Environment=\"OLLAMA_NUM_PARALLEL=${SAFETY_NUM_PARALLEL}\"" \
    "Environment=\"OLLAMA_SCHED_SPREAD=${SAFETY_SCHED_SPREAD}\"" \
    "Environment=\"OLLAMA_CONTEXT_LENGTH=${SAFETY_CONTEXT_LENGTH}\"" \
    "Environment=\"OLLAMA_KEEP_ALIVE=${SAFETY_KEEP_ALIVE}\"" \
    "Environment=\"OLLAMA_MAX_QUEUE=${SAFETY_MAX_QUEUE}\"" \
    "Environment=\"OLLAMA_FLASH_ATTENTION=1\"" \
    "Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\""
  if [ "$SAFETY_VRAM_RESERVE_MIB" -gt 0 ]; then
    printf '%s\n' "Environment=\"OLLAMA_GPU_OVERHEAD=${SAFETY_VRAM_RESERVE_BYTES}\""
  fi
  if [ "$SAFETY_GPU_STRICT" = 1 ]; then
    printf '%s\n' \
      "Environment=\"LLAMA_ARG_N_GPU_LAYERS=all\"" \
      "Environment=\"LLAMA_ARG_SPLIT_MODE=layer\"" \
      "Environment=\"LLAMA_ARG_FIT=off\"" \
      "Environment=\"GGML_CUDA_NO_PINNED=1\""
  fi
}

render_safety_shell_exports() {
  if [ "$SAFETY_GPU_STRICT" = 1 ]; then
    printf '%s\n' "unset GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_REGISTER_HOST LLAMA_ARG_FIT_TARGET"
  else
    printf '%s\n' "unset GGML_CUDA_NO_PINNED LLAMA_ARG_N_GPU_LAYERS LLAMA_ARG_SPLIT_MODE LLAMA_ARG_FIT LLAMA_ARG_FIT_TARGET"
  fi
  if [ "$SAFETY_VRAM_RESERVE_MIB" -eq 0 ]; then
    printf '%s\n' "unset OLLAMA_GPU_OVERHEAD"
  fi
  if [ "$SAFETY_BACKEND" = "rocm" ]; then
    printf '%s\n' "unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL"
  fi
  local line assignment
  while IFS= read -r line; do
    [[ "$line" == Environment=\"*\" ]] || continue
    assignment="${line#Environment=\"}"
    assignment="${assignment%\"}"
    printf 'export %q\n' "$assignment"
  done < <(render_safety_environment_directives)
}

render_safety_preflight_script() {
  cat <<'PREFLIGHT'
#!/bin/sh
# Generated by ollama-unify. Refuse to start Ollama while the host is already
# short of memory or stalled in reclaim. On modern systemd this runs as an
# ExecCondition, so a refusal skips startup without marking the unit failed.
set -eu

reserve_mib=${1:-}
pressure_limit=${2:-}
case "$reserve_mib:$pressure_limit" in
  *[!0-9:]*|:|*:)
    echo "ollama-unify preflight: invalid reserve or pressure limit" >&2
    exit 75
    ;;
esac

mem_available_kib=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo 2>/dev/null || true)
case "$mem_available_kib" in
  ''|*[!0-9]*)
    echo "ollama-unify preflight: cannot read MemAvailable" >&2
    exit 75
    ;;
esac
mem_available_mib=$((mem_available_kib / 1024))
if [ "$mem_available_mib" -lt "$reserve_mib" ]; then
  echo "ollama-unify preflight: refusing start; ${mem_available_mib} MiB available, ${reserve_mib} MiB reserved" >&2
  exit 75
fi

if [ -r /proc/pressure/memory ]; then
  full_avg10=$(awk '/^full / { for (i=1; i<=NF; i++) if ($i ~ /^avg10=/) { sub(/^avg10=/, "", $i); print $i; exit } }' /proc/pressure/memory)
  if [ -n "$full_avg10" ] && awk -v actual="$full_avg10" -v limit="$pressure_limit" 'BEGIN { exit !(actual >= limit) }'; then
    echo "ollama-unify preflight: refusing start; memory full avg10=${full_avg10}% (limit ${pressure_limit}%)" >&2
    exit 75
  fi
fi
exit 0
PREFLIGHT
}

render_safety_service_directives() {
  if [ "$SYSTEMD_VERSION" -ge 235 ]; then
    printf '%s\n' "UnsetEnvironment=OLLAMA_LLM_LIBRARY"
    if [ "$SAFETY_VRAM_RESERVE_MIB" -eq 0 ]; then
      printf '%s\n' "UnsetEnvironment=OLLAMA_GPU_OVERHEAD"
    fi
    if [ "$SAFETY_GPU_STRICT" = 1 ]; then
      # llama.cpp treats mere presence as true for these CUDA flags. They must
      # be absent, not assigned the string "0".
      printf '%s\n' "UnsetEnvironment=GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_REGISTER_HOST LLAMA_ARG_FIT_TARGET"
    else
      printf '%s\n' "UnsetEnvironment=GGML_CUDA_NO_PINNED LLAMA_ARG_N_GPU_LAYERS LLAMA_ARG_SPLIT_MODE LLAMA_ARG_FIT LLAMA_ARG_FIT_TARGET"
    fi
  fi
  if [ "$SAFETY_BACKEND" = "rocm" ] && [ "$SYSTEMD_VERSION" -ge 235 ]; then
    printf '%s\n' "UnsetEnvironment=CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL"
  fi
  render_safety_environment_directives
  if [ "$SYSTEMD_VERSION" -ge 231 ]; then
    printf '%s\n' "MemoryAccounting=yes" "MemoryHigh=${SAFETY_HOST_MEMORY_HIGH_MIB}M" \
      "MemoryMax=${SAFETY_HOST_MEMORY_MAX_MIB}M"
  fi
  if [ "$SYSTEMD_VERSION" -ge 232 ]; then printf '%s\n' "MemorySwapMax=${SAFETY_SWAP_MAX}"; fi
  if [ "$SYSTEMD_VERSION" -ge 243 ]; then printf '%s\n' "OOMPolicy=stop"; fi
  if [ "$SYSTEMD_VERSION" -ge 227 ]; then
    printf '%s\n' "CPUAccounting=yes" "CPUWeight=${SAFETY_CPU_WEIGHT}" \
      "CPUQuota=${SAFETY_CPU_QUOTA_PERCENT}%" "IOAccounting=yes" "IOWeight=${SAFETY_IO_WEIGHT}"
  fi
  if [ "$SYSTEMD_VERSION" -ge 247 ]; then
    printf '%s\n' "ManagedOOMMemoryPressure=kill" \
      "ManagedOOMMemoryPressureLimit=${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}%" \
      "ManagedOOMSwap=kill"
  fi
  printf '%s\n' "OOMScoreAdjust=750" "Nice=10" "KillMode=control-group"
  if [ "$SYSTEMD_VERSION" -ge 243 ]; then
    printf '%s\n' "ExecCondition=${SAFETY_PREFLIGHT_PATH} ${SAFETY_HOST_RESERVE_MIB} ${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}"
  else
    printf '%s\n' "ExecStartPre=${SAFETY_PREFLIGHT_PATH} ${SAFETY_HOST_RESERVE_MIB} ${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}"
  fi
  printf '%s\n' "Restart=${SAFETY_RESTART_POLICY}" "RestartSec=60s"
  if [ ${#SAFETY_PREFLIGHT_DIRECTIVES[@]} -gt 0 ]; then printf '%s\n' "${SAFETY_PREFLIGHT_DIRECTIVES[@]}"; fi
}

install_systemd_safety_policy() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  local dropin_dir="/etc/systemd/system/ollama.service.d"
  local safety_file="$dropin_dir/zzz-ollama-unify-safety.conf"
  local preflight_dir="${SAFETY_PREFLIGHT_PATH%/*}"

  "${elevate[@]}" mkdir -p "$dropin_dir" "$preflight_dir"
  render_safety_preflight_script | "${elevate[@]}" tee "$SAFETY_PREFLIGHT_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_PREFLIGHT_PATH"
  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf '[Unit]\nStartLimitIntervalSec=5min\nStartLimitBurst=2\n\n[Service]\n'
    render_safety_service_directives
  } | "${elevate[@]}" tee "$safety_file" >/dev/null

  "${elevate[@]}" systemctl daemon-reload
  if [ "$SYSTEMD_VERSION" -ge 247 ]; then
    if "${elevate[@]}" systemctl enable --now systemd-oomd.service >/dev/null 2>&1; then
      ok "systemd-oomd is enabled for proactive memory-pressure kills"
    else
      warn "systemd-oomd could not be enabled; MemoryMax remains active, but PSI-based killing is unavailable"
    fi
  fi
  if command -v systemd-analyze >/dev/null 2>&1; then
    "${elevate[@]}" systemd-analyze verify ollama.service
  fi
  ok "memory-pressure preflight installed: $SAFETY_PREFLIGHT_PATH"
  ok "late-priority safety drop-in installed: $safety_file"
}

install_safety_only() {
  banner
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--install-safety requires a systemd host"; exit 2; }
  systemctl cat ollama.service >/dev/null 2>&1 \
    || { err "ollama.service was not found"; exit 2; }
  require awk
  build_safety_profile
  print_safety_profile

  local SUDO=""
  if [ "$EUID" -ne 0 ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo is required to install the systemd policy"; exit 2; }
    SUDO="sudo"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  local was_active=0
  if systemctl is-active ollama.service >/dev/null 2>&1; then
    was_active=1
    $SUDO systemctl stop ollama.service
    ok "ollama.service stopped while its policy is replaced"
  fi

  hdr "Installing Ollama safety policy"
  install_systemd_safety_policy "$SUDO"

  if [ "$was_active" = 1 ]; then
    $SUDO systemctl start ollama.service || true
    if systemctl is-active ollama.service >/dev/null 2>&1; then
      ok "ollama.service restarted under the new policy"
    else
      warn "ollama.service remains stopped because the safety condition refused the restart"
    fi
  else
    say "  ollama.service was already stopped and has been left stopped."
  fi
}

preview_safety_profile() {
  banner
  build_safety_profile
  print_safety_profile
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    hdr "Generated late-priority systemd drop-in"
    printf '%s\n' "# Managed by ollama-unify" "[Unit]" "StartLimitIntervalSec=5min" \
      "StartLimitBurst=2" "" "[Service]"
    render_safety_service_directives
  else
    hdr "Generated portable Ollama environment policy"
    render_safety_shell_exports
  fi
}

classify_host() {
  banner
  build_safety_profile
  print_safety_profile
}

print_environment_policy() {
  build_safety_profile
  render_safety_shell_exports
}

# ─────────────────────────────────────────── store discovery (read-only)
declare -a STORE_PATHS=()
canonical_path() {
  local path="$1"
  if command -v realpath >/dev/null 2>&1 && realpath -m -- "$path" >/dev/null 2>&1; then
    realpath -m -- "$path"
    return
  fi
  if readlink -m -- "$path" >/dev/null 2>&1; then
    readlink -m -- "$path"
    return
  fi
  if [ -d "$path" ]; then
    (cd "$path" 2>/dev/null && pwd -P)
    return
  fi
  [[ "$path" == /* ]] || path="$PWD/$path"
  local parent="${path%/*}" base="${path##*/}"
  if [ -d "$parent" ]; then
    printf '%s/%s\n' "$(cd "$parent" 2>/dev/null && pwd -P)" "$base"
  else
    printf '%s\n' "$path"
  fi
}

add_store() {
  local p="$1"
  [ -n "$p" ] || return 0
  # canonicalize without requiring existence
  p="$(canonical_path "$p")"
  [ -d "$p" ] || return 0
  for existing in "${STORE_PATHS[@]:-}"; do [ "$existing" = "$p" ] && return 0; done
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
    v=$(grep -E '^OLLAMA_MODELS=' /etc/default/ollama 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'' || true)
    add_store "$v"
  fi
  # /etc/environment
  if [ -r /etc/environment ]; then
    local v
    v=$(grep -E '^OLLAMA_MODELS=' /etc/environment 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'' || true)
    add_store "$v"
  fi
  # systemd unit + drop-ins
  if [ "$HAS_SYSTEMD" = 1 ]; then
    if systemctl cat ollama >/dev/null 2>&1; then
      local v
      v=$(systemctl cat ollama 2>/dev/null \
        | sed -n 's/.*OLLAMA_MODELS=\([^"[:space:]]*\).*/\1/p' | tail -1)
      add_store "$v"
    fi
  fi
  # running runner cmdlines
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r path; do add_store "$path"; done < <(
      pgrep -af 'ollama runner' 2>/dev/null \
        | sed -n 's#.*\(/[^ ]*/blobs/sha256-[a-f0-9]*\).*#\1#p' \
        | sed 's|/blobs/.*||' | sort -u
    )
  fi
}

count_manifests() { find "$1/manifests" -type f 2>/dev/null | wc -l; }
count_blobs()     { find "$1/blobs"     -type f 2>/dev/null | wc -l; }
fs_of()           { df -Pk "$1" 2>/dev/null | awk 'END {print $1" ("$NF")"}'; }
dev_of()          { df -Pk "$1" 2>/dev/null | awk 'END {print $1}'; }
mount_of()        { df -Pk "$1" 2>/dev/null | awk 'END {print $NF}'; }
own_of()          { stat -c '%U:%G' "$1" 2>/dev/null || stat -f '%Su:%Sg' "$1" 2>/dev/null || echo "?"; }
size_of() {
  if du -sb /dev/null >/dev/null 2>&1; then du -sb "$1" 2>/dev/null | awk '{print $1}'
  else du -sk "$1" 2>/dev/null | awk '{print $1 * 1024}'; fi
}
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
  df -hP 2>/dev/null \
    | awk 'NR==1 {print "  "$0; next} ($NF ~ /^\/(home|Users|srv|var|opt|mnt|media|data|Volumes)(\/|$)/) || $NF=="/" {if (!seen[$NF]++) print "  "$0}'
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
  # Reflink test when GNU cp exposes it; otherwise use rsync cross-filesystem.
  local probe="$dst/.reflink_probe_$$" cp_help=""
  cp_help=$(cp --help 2>&1 || true)
  if [[ "$cp_help" == *--reflink* ]] && cp --reflink=always /dev/null "$probe" 2>/dev/null; then
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
        while IFS= read -r -d '' item; do
          "${elevate[@]}" mv -n "$item" "$dst/manifests/"
        done < <(find "$src/manifests" -mindepth 1 -maxdepth 1 -print0)
      fi
      if [ -d "$src/blobs" ]; then
        while IFS= read -r -d '' item; do
          "${elevate[@]}" mv -n "$item" "$dst/blobs/"
        done < <(find "$src/blobs" -mindepth 1 -maxdepth 1 -print0)
      fi
      ;;
    reflink)
      ok "reflink-capable filesystem detected → using cp --reflink=auto"
      "${elevate[@]}" cp -a --reflink=auto -n "$src/." "$dst/"
      ;;
    rsync)
      ok "cross-filesystem copy → using rsync (local-copy tuned)"
      local rsync_help
      rsync_help=$(rsync --help 2>&1)
      local -a rsync_args=(-a --ignore-existing)
      [[ "$rsync_help" == *--hard-links* ]] && rsync_args+=(-H)
      [[ "$rsync_help" == *--whole-file* ]] && rsync_args+=(--whole-file)
      [[ "$rsync_help" == *--inplace* ]] && rsync_args+=(--inplace)
      [[ "$rsync_help" == *--no-compress* ]] && rsync_args+=(--no-compress)
      if [[ "$rsync_help" == *--info* ]]; then rsync_args+=(--info=progress2); else rsync_args+=(--progress); fi
      "${elevate[@]}" rsync "${rsync_args[@]}" "$src/" "$dst/"
      ;;
  esac
}

plan_requires_sudo() {
  local do_systemd="$1" do_service_user="$2" destination="$3"
  if [ "$do_systemd" = 1 ] || [ "$do_service_user" = 1 ] \
    || [[ "$destination" =~ ^/(srv|var|opt|usr|etc) ]]; then
    return 0
  fi
  [ ${#SERVICE_PIDS[@]} -gt 0 ] && [ -n "${SERVICE_PIDS[0]:-}" ]
}

# ───────────────────────────────────────────────────────────── main flow
main() {
  case "${1:-}" in
    --classify)
      classify_host
      exit 0
      ;;
    --safety-preview)
      preview_safety_profile
      exit 0
      ;;
    --print-env)
      print_environment_policy
      exit 0
      ;;
    --install-safety)
      detect_host_profile
      install_safety_only
      exit 0
      ;;
    -h|--help)
      say "Usage: ./ollama-unify.sh [--classify|--safety-preview|--print-env|--install-safety]"
      say "  --classify        Classify the host, accelerators, backend, and risk tier without changes."
      say "  --safety-preview  Classify the host and print the generated safety policy without changes."
      say "  --print-env       Print shell exports for the selected scheduler/backend policy."
      say "  --install-safety  Install only the fail-closed systemd safety policy; do not migrate models."
      exit 0
      ;;
    "") ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac

  require_migration_tools
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

  build_safety_profile
  print_safety_profile

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
      free=$(df -Pk "$s" 2>/dev/null | awk 'END {print $4}')
      [ -z "$free" ] && continue
      if [ "$free" -gt "$max_free" ]; then max_free=$free; default_dst="$s"; fi
    done

    hdr "Destination"
    DEST=$(ask "Where to unify all models?" "$default_dst")
    DEST="$(canonical_path "$DEST")"
  fi

  # opt-ins
  hdr "Options"
  local DO_SYSTEMD=0 DO_SYSTEMD_MODELS=0 DO_SAFETY=0 DO_SYMLINKS=0 DO_BASHRC=0 DO_SERVICE_USER=0
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl cat ollama >/dev/null 2>&1; then
    if confirm "Update ollama.service via drop-in to use OLLAMA_MODELS=$DEST?" "Y"; then
      DO_SYSTEMD=1
      DO_SYSTEMD_MODELS=1
    fi

    if confirm "Apply this classified backend and OOM safety policy to ollama.service?" "Y"; then
      DO_SYSTEMD=1
      DO_SAFETY=1
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
  [ "$DO_SAFETY"       = 1 ] && say "  • Install fail-closed memory-pressure, CPU, I/O, and OOM guardrails"
  [ "$DO_SERVICE_USER" = 1 ] && say "  • Change service User/Group to $USER"
  [ "$DO_SYMLINKS"     = 1 ] && say "  • Symlink originals → destination"
  [ "$DO_BASHRC"       = 1 ] && say "  • Add OLLAMA_MODELS export to $SHELL_RC"
  say "  • Stop running daemons, perform transfer, restart ollama.service"

  echo
  confirm "Proceed?" "N" || { warn "Aborted."; exit 1; }

  # ── sudo gate
  local SUDO=""
  if plan_requires_sudo "$DO_SYSTEMD" "$DO_SERVICE_USER" "$DEST"; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo required but not installed."; exit 2; }
    SUDO="sudo"
    say "${C_DIM}(sudo will prompt for your password)${C_RST}"
    $SUDO -n true 2>/dev/null || $SUDO -v
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
      printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
      printf '[Service]\n'
      if [ "$DO_SYSTEMD_MODELS" = 1 ]; then
        printf 'Environment="OLLAMA_MODELS=%s"\n' "$DEST"
      fi
      if [ "$DO_SERVICE_USER" = 1 ]; then
        printf 'User=%s\n' "$USER"
        printf 'Group=%s\n' "$(id -gn)"
      fi
    } | $SUDO tee "$dropin_file" >/dev/null
    ok "late-priority drop-in installed: $dropin_file"
    if [ "$DO_SAFETY" = 1 ]; then
      install_systemd_safety_policy "$SUDO"
    else
      $SUDO systemctl daemon-reload
      if command -v systemd-analyze >/dev/null 2>&1; then
        $SUDO systemd-analyze verify ollama.service
      fi
    fi
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
    if $SUDO systemctl start ollama; then
      sleep 3
      if systemctl is-active ollama >/dev/null 2>&1; then
        ok "ollama.service is active"
      else
        local service_result
        service_result=$(systemctl show ollama.service -p Result --value 2>/dev/null || true)
        if [ "$service_result" = "exec-condition" ]; then
          warn "ollama.service was safely held inactive by the memory-pressure condition"
        else
          err "ollama.service stopped after launch — check 'journalctl -u ollama -n 50'"
        fi
      fi
    else
      warn "ollama.service was left stopped; the safety preflight may have refused an unsafe restart"
      warn "check: journalctl -u ollama -n 50"
    fi
  fi

  hdr "Verification"
  local mcount; mcount=$(count_manifests "$DEST")
  ok "$mcount manifests at $DEST"
  if [ "$HAS_CURL" = 1 ]; then
    local api_count
    api_count=$(curl -s --max-time 5 http://127.0.0.1:11434/api/tags 2>/dev/null \
      | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | wc -l || true)
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
