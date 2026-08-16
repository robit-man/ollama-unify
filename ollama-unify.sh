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
SAFETY_GPU_PREFERRED=0
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
SAFETY_NEGOTIATOR_ENABLED=0
SAFETY_NEGOTIATOR_PATH="/usr/local/libexec/ollama-unify-gpu-negotiator"
SAFETY_NEGOTIATOR_CLI_PATH="/usr/local/bin/ollama-unify-gpu-lease"
SAFETY_NEGOTIATOR_CONFIG_PATH="/etc/default/ollama-unify-negotiator"
SAFETY_NEGOTIATOR_UNIT_PATH="/etc/systemd/system/ollama-unify-negotiator.service"
SAFETY_NEGOTIATOR_SOCKET="/run/ollama-unify/gpu-negotiator.sock"
SAFETY_OLLAMA_BACKEND="127.0.0.1:11436"
SAFETY_DOCKER_PLUGIN_PATH="/usr/local/lib/docker/cli-plugins/docker-gpu"
SAFETY_LEGACY_DOCKER_PLUGIN_PATH="/usr/local/lib/docker/cli-plugins/docker-gpu-lease"
SAFETY_DISCOVERY_DIR="/usr/local/share/ollama-unify"
SAFETY_DISCOVERY_PATH="/usr/local/share/ollama-unify/gpu-negotiator.json"
SAFETY_AGENT_INSTRUCTIONS_PATH="/usr/local/share/ollama-unify/AGENTS.md"
SAFETY_STATE_PATH="/usr/local/share/ollama-unify/state.env"
SAFETY_RECONCILE_HELPER_PATH="/usr/local/libexec/ollama-unify-reconcile"
SAFETY_RECONCILE_SERVICE_PATH="/etc/systemd/system/ollama-unify-reconcile.service"
SAFETY_RECONCILE_PATH_UNIT_PATH="/etc/systemd/system/ollama-unify-reconcile.path"
SAFETY_OLLAMA_RELEASE_API="https://api.github.com/repos/ollama/ollama/releases/latest"
SAFETY_OLLAMA_INSTALL_URL="https://ollama.com/install.sh"

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

  SAFETY_GPU_PREFERRED=0
  # Do not force every selected accelerator to participate in every load.
  # Ollama's native scheduler can then place against live free VRAM and only
  # split a model when the load actually requires multiple devices.
  SAFETY_SCHED_SPREAD=0
  if [[ "$SAFETY_BACKEND" =~ ^(cuda|rocm)$ ]] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ]; then
    SAFETY_GPU_PREFERRED=1
  fi
  if [ "$SAFETY_GPU_PREFERRED" = 1 ] && [ "$HOST_OS" = "Linux" ] \
    && [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    SAFETY_NEGOTIATOR_ENABLED=1
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
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-0}"
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
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
    say "  GPU policy: native live-VRAM placement; forced spread disabled; pageable/cgroup-bounded CPU overflow only"
    say "  GPU host paths: unified spill and pinned-host buffers disabled"
  fi
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    say "  GPU negotiator: cooperative leases plus anonymous-process rebalance; Ollama refits after external allocation"
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
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
    printf '%s\n' \
      "Environment=\"LLAMA_ARG_N_GPU_LAYERS=auto\"" \
      "Environment=\"LLAMA_ARG_SPLIT_MODE=layer\"" \
      "Environment=\"LLAMA_ARG_FIT=on\"" \
      "Environment=\"GGML_CUDA_NO_PINNED=1\""
  fi
}

render_safety_shell_exports() {
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
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

render_gpu_negotiator_script() {
  cat <<'NEGOTIATOR'
#!/usr/bin/env python3
# Generated by ollama-unify. This process owns Ollama's public HTTP port,
# drains in-flight requests around cooperative GPU leases, unloads resident
# runners, and lets Ollama refit from live VRAM after external allocation.
from __future__ import annotations

import argparse
import http.client
import http.server
import json
import logging
import os
import secrets
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        return value if value >= 0 else default
    except ValueError:
        return default


def load_environment_file(path: str) -> None:
    """Load the installer's simple quoted KEY=VALUE file for standalone CLI calls."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("OLLAMA_UNIFY_") or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def split_address(value: str) -> tuple[str, int]:
    value = value.removeprefix("http://").removeprefix("https://").rstrip("/")
    if value.startswith("[") and "]:" in value:
        host, port = value[1:].split("]:", 1)
        return host, int(port)
    if ":" not in value:
        return value, 11434
    host, port = value.rsplit(":", 1)
    return host, int(port)


load_environment_file(os.environ.get(
    "OLLAMA_UNIFY_CONFIG", "/etc/default/ollama-unify-negotiator"
))
BACKEND_HOST, BACKEND_PORT = split_address(os.environ.get("OLLAMA_UNIFY_BACKEND", "127.0.0.1:11436"))
LISTEN_HOST, LISTEN_PORT = split_address(os.environ.get("OLLAMA_UNIFY_LISTEN", "127.0.0.1:11434"))
CONTROL_SOCKET = os.environ.get("OLLAMA_UNIFY_SOCKET", "/run/ollama-unify/gpu-negotiator.sock")
DRAIN_TIMEOUT = env_float("OLLAMA_UNIFY_DRAIN_TIMEOUT", 300.0)
UNLOAD_TIMEOUT = env_float("OLLAMA_UNIFY_UNLOAD_TIMEOUT", 120.0)
DEFAULT_LEASE_TTL = env_int("OLLAMA_UNIFY_LEASE_TTL", 300)
MAX_CONTEXT = env_int("OLLAMA_UNIFY_MAX_CONTEXT", 0)
ANON_POLL = env_float("OLLAMA_UNIFY_ANON_POLL", 0.5)
ANON_SETTLE = env_float("OLLAMA_UNIFY_ANON_SETTLE", 2.0)
ANON_MAX_DRAIN = env_float("OLLAMA_UNIFY_ANON_MAX_DRAIN", 15.0)
BACKEND_TYPE = os.environ.get("OLLAMA_UNIFY_BACKEND_TYPE", "unknown")
SELECTED_GPUS = [value for value in os.environ.get("OLLAMA_UNIFY_SELECTED_GPUS", "").split(",") if value]
LOG = logging.getLogger("ollama-unify-negotiator")

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
NATIVE_MODEL_PATHS = (
    "/api/generate", "/api/chat", "/api/embed", "/api/embeddings", "/api/rerank",
)


def clamp_request(path: str, content_type: str, body: bytes) -> bytes:
    """Prevent clients from bypassing dynamic GPU fitting or the scanned context cap."""
    if not body or not path.startswith(NATIVE_MODEL_PATHS) or "json" not in content_type.lower():
        return body
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body
    if not isinstance(payload, dict):
        return body
    options = payload.get("options")
    if options is None:
        options = {}
        payload["options"] = options
    if isinstance(options, dict):
        options["num_gpu"] = -1
        options.pop("main_gpu", None)
        if MAX_CONTEXT > 0:
            requested = options.get("num_ctx")
            if isinstance(requested, (int, float)) and requested > MAX_CONTEXT:
                options["num_ctx"] = MAX_CONTEXT
    payload.pop("num_gpu", None)
    payload.pop("main_gpu", None)
    return json.dumps(payload, separators=(",", ":")).encode()


def backend_json(method: str, path: str, payload: dict[str, Any] | None = None,
                 timeout: float = 10.0) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        if response.status >= 400:
            raise RuntimeError(f"Ollama backend {method} {path} returned {response.status}: {data[:300]!r}")
        return json.loads(data or b"{}")
    finally:
        conn.close()


def running_models() -> list[dict[str, Any]]:
    try:
        models = backend_json("GET", "/api/ps", timeout=3.0).get("models", [])
        return models if isinstance(models, list) else []
    except (OSError, RuntimeError, ValueError):
        return []


def unload_all_models(timeout: float = UNLOAD_TIMEOUT) -> list[str]:
    models = running_models()
    names = [str(model.get("name") or model.get("model") or "") for model in models]
    names = [name for name in names if name]
    for name in names:
        try:
            backend_json("POST", "/api/generate", {
                "model": name, "keep_alive": 0, "stream": False,
            }, timeout=min(timeout, 30.0))
        except (OSError, RuntimeError, ValueError) as exc:
            LOG.warning("unload request failed for %s: %s", name, exc)

    deadline = time.monotonic() + timeout
    while names and time.monotonic() < deadline:
        if not running_models():
            return names
        time.sleep(0.2)
    if names and running_models():
        raise TimeoutError(f"Ollama models did not unload within {timeout:.0f}s")
    return names


def gpu_snapshot() -> list[dict[str, Any]]:
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    devices = []
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            continue
        try:
            devices.append({
                "uuid": fields[0], "total_mib": int(fields[1]),
                "used_mib": int(fields[2]), "free_mib": int(fields[3]),
            })
        except ValueError:
            continue
    return devices


def ollama_cgroup_pids() -> set[int]:
    try:
        result = subprocess.run(
            ["systemctl", "show", "ollama.service", "-p", "ControlGroup", "--value"],
            check=True, capture_output=True, text=True, timeout=3,
        )
        control_group = result.stdout.strip().lstrip("/")
        path = Path("/sys/fs/cgroup") / control_group / "cgroup.procs"
        return {int(line) for line in path.read_text().splitlines() if line.isdigit()}
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        return set()


def foreign_gpu_usage() -> dict[str, int]:
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    ollama_pids = ollama_cgroup_pids()
    usage: dict[str, int] = {}
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            used = int(fields[2])
        except ValueError:
            continue
        if pid not in ollama_pids:
            usage[f"{pid}@{fields[1]}"] = used
    return usage


def host_memory_snapshot() -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemAvailable:", "MemTotal:")):
                key, value, *_ = line.split()
                result[key.rstrip(":").lower() + "_mib"] = int(value) // 1024
    except (OSError, ValueError):
        pass
    try:
        control_group = subprocess.run(
            ["systemctl", "show", "ollama.service", "-p", "ControlGroup", "--value"],
            check=True, capture_output=True, text=True, timeout=3,
        ).stdout.strip().lstrip("/")
        base = Path("/sys/fs/cgroup") / control_group
        for source, target in (("memory.current", "ollama_current_bytes"),
                               ("memory.high", "ollama_high_bytes"),
                               ("memory.max", "ollama_max_bytes")):
            value = (base / source).read_text().strip()
            result[target] = int(value) if value.isdigit() else value
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


def discovery_document() -> dict[str, Any]:
    devices = gpu_snapshot()
    selected = set(SELECTED_GPUS)
    for device in devices:
        device["selected_for_ollama"] = not selected or device.get("uuid") in selected
    return {
        "schema": "io.ollama-unify.gpu-negotiator.discovery.v1",
        "protocol": "ollama-unify-gpu-lease/v1",
        "available": True,
        "backend": BACKEND_TYPE,
        "selected_gpu_ids": SELECTED_GPUS,
        "selected_gpu_count": len(SELECTED_GPUS),
        "gpus": devices,
        "public_ollama_api": f"http://127.0.0.1:{LISTEN_PORT}",
        "ollama_backend": f"http://{BACKEND_HOST}:{BACKEND_PORT}",
        "control_socket": CONTROL_SOCKET,
        "discovery_file": "/usr/local/share/ollama-unify/gpu-negotiator.json",
        "agent_instructions": "/usr/local/share/ollama-unify/AGENTS.md",
        "well_known": f"http://127.0.0.1:{LISTEN_PORT}/.well-known/ollama-unify-gpu-negotiator",
        "commands": {
            "discover": "docker gpu discover",
            "status": "docker gpu status",
            "cooperative_run": (
                "docker gpu run --owner NAME --vram-mib MIB "
                "--ready-command 'READINESS_CHECK' -- COMMAND"
            ),
            "manual": ["acquire", "ready", "prepare", "release", "heartbeat"],
        },
        "requirements": {
            "cuda_deployments": (
                "Acquire a lease before loading CUDA models; signal ready only after GPU allocation is resident."
            ),
            "resize": "Call prepare before increasing VRAM use, then ready after the new allocation settles.",
            "release": "Free external CUDA allocations before releasing the lease.",
            "num_gpu_semantics": (
                "Ollama num_gpu is the number of GPU-offloaded layers, not the physical GPU count; -1 is automatic."
            ),
            "anonymous_cuda": "Reactive best-effort only; cooperative leases are required for OOM prevention.",
        },
    }


def agent_instructions_text() -> str:
    return """# Host CUDA negotiation\n\nThis host runs the ollama-unify GPU lease broker. Before creating, starting, or resizing any Docker/container/service deployment that uses CUDA:\n\n1. Run `docker gpu discover` and inspect the selected GPUs and current policy.\n2. Launch long-running CUDA services with `docker gpu run --owner NAME --vram-mib MIB --ready-command 'CHECK' -- COMMAND`. The readiness check must pass only after CUDA models are resident.\n3. For an independently supervised service, use `acquire`, load the CUDA models, call `ready`, call `prepare` before any VRAM growth, and call `release` only after freeing CUDA memory.\n4. Never bypass the broker by assuming free VRAM from a static scan. Anonymous allocation is reactive and cannot prevent the first CUDA OOM.\n5. Ollama API `num_gpu` counts offloaded layers, not physical GPUs. Keep it automatic (`-1`); every selected GPU remains available to the scheduler.\n\nMachine-readable discovery: `/usr/local/share/ollama-unify/gpu-negotiator.json` or `http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator`.\n"""


def wait_for_foreign_settle() -> None:
    deadline = time.monotonic() + ANON_MAX_DRAIN
    last = foreign_gpu_usage()
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(ANON_POLL)
        current = foreign_gpu_usage()
        if current == last:
            if time.monotonic() - stable_since >= ANON_SETTLE:
                return
        else:
            last = current
            stable_since = time.monotonic()


@dataclass
class Lease:
    token: str
    owner: str
    state: str
    requested_mib: int
    created_at: float
    heartbeat_at: float
    ttl: int


class Broker:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.transition = threading.Lock()
        self.draining = False
        self.active_requests = 0
        self.leases: dict[str, Lease] = {}
        self.last_reason = "startup"
        self.stopping = threading.Event()
        self.anonymous_running = False

    def proxy_enter(self) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT
        with self.cv:
            while self.draining:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("GPU negotiation is still draining Ollama")
                self.cv.wait(min(remaining, 1.0))
            self.active_requests += 1

    def proxy_exit(self) -> None:
        with self.cv:
            self.active_requests = max(0, self.active_requests - 1)
            self.cv.notify_all()

    def begin_drain(self, reason: str) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT
        with self.cv:
            self.draining = True
            self.last_reason = reason
            self.cv.notify_all()
            while self.active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.draining = False
                    self.cv.notify_all()
                    raise TimeoutError(f"{self.active_requests} Ollama request(s) did not drain")
                self.cv.wait(min(remaining, 1.0))

    def end_drain(self) -> None:
        with self.cv:
            self.draining = False
            self.cv.notify_all()

    def pending_lease(self) -> bool:
        return any(lease.state == "pending" for lease in self.leases.values())

    def acquire(self, owner: str, requested_mib: int, ttl: int) -> dict[str, Any]:
        with self.transition:
            with self.cv:
                if self.pending_lease():
                    raise RuntimeError("another lease is waiting for its external workload to become ready")
            self.begin_drain(f"lease acquire by {owner}")
            try:
                unloaded = unload_all_models()
                devices = gpu_snapshot()
                aggregate_free = sum(int(device["free_mib"]) for device in devices)
                if requested_mib > 0 and devices and requested_mib > aggregate_free:
                    raise RuntimeError(
                        f"requested {requested_mib} MiB but only {aggregate_free} MiB is free after Ollama unload"
                    )
                now = time.time()
                token = secrets.token_urlsafe(24)
                lease = Lease(token, owner, "pending", requested_mib, now, now, ttl)
                with self.cv:
                    self.leases[token] = lease
                LOG.info("lease acquired owner=%s requested_mib=%s unloaded=%s", owner, requested_mib, unloaded)
                return {"ok": True, "lease": asdict(lease), "unloaded": unloaded,
                        "gpus": devices, "host_memory": host_memory_snapshot()}
            except Exception:
                self.end_drain()
                raise

    def ready(self, token: str) -> dict[str, Any]:
        with self.transition:
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                if lease.state != "pending":
                    raise RuntimeError("lease is not pending")
                lease.state = "active"
                lease.heartbeat_at = time.time()
            devices = gpu_snapshot()
            self.end_drain()
            LOG.info("lease ready owner=%s", lease.owner)
            return {"ok": True, "lease": asdict(lease), "gpus": devices,
                    "host_memory": host_memory_snapshot()}

    def prepare(self, token: str) -> dict[str, Any]:
        with self.transition:
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                if self.pending_lease():
                    raise RuntimeError("a lease transition is already pending")
            self.begin_drain(f"lease resize by {lease.owner}")
            try:
                unloaded = unload_all_models()
                with self.cv:
                    lease.state = "pending"
                    lease.heartbeat_at = time.time()
                return {"ok": True, "lease": asdict(lease), "unloaded": unloaded,
                        "gpus": gpu_snapshot()}
            except Exception:
                self.end_drain()
                raise

    def release(self, token: str, reason: str = "lease release") -> dict[str, Any]:
        with self.transition:
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                already_drained = lease.state == "pending" and self.draining
            if not already_drained:
                self.begin_drain(f"{reason} by {lease.owner}")
            try:
                unloaded = unload_all_models()
                wait_for_foreign_settle()
                with self.cv:
                    self.leases.pop(token, None)
                LOG.info("lease released owner=%s reason=%s", lease.owner, reason)
                return {"ok": True, "released": token, "unloaded": unloaded,
                        "gpus": gpu_snapshot(), "host_memory": host_memory_snapshot()}
            finally:
                self.end_drain()

    def heartbeat(self, token: str) -> dict[str, Any]:
        with self.cv:
            lease = self.leases.get(token)
            if lease is None:
                raise KeyError("unknown lease")
            lease.heartbeat_at = time.time()
            return {"ok": True, "lease": asdict(lease)}

    def status(self) -> dict[str, Any]:
        with self.cv:
            leases = [asdict(lease) for lease in self.leases.values()]
            draining = self.draining
            active = self.active_requests
            reason = self.last_reason
        return {"ok": True, "draining": draining, "active_requests": active,
                "last_reason": reason, "leases": leases, "gpus": gpu_snapshot(),
                "foreign_gpu_processes": foreign_gpu_usage(), "models": running_models(),
                "host_memory": host_memory_snapshot()}

    def anonymous_rebalance(self, reason: str) -> None:
        if not self.transition.acquire(blocking=False):
            with self.cv:
                self.anonymous_running = False
            return
        try:
            self.begin_drain(reason)
            unload_all_models()
            wait_for_foreign_settle()
            LOG.warning("reactive anonymous GPU rebalance completed: %s", reason)
        except Exception as exc:
            LOG.error("anonymous GPU rebalance failed: %s", exc)
        finally:
            self.end_drain()
            self.transition.release()
            with self.cv:
                self.anonymous_running = False

    def anonymous_watcher(self) -> None:
        previous = foreign_gpu_usage()
        while not self.stopping.wait(ANON_POLL):
            current = foreign_gpu_usage()
            changed = set(current) != set(previous)
            with self.cv:
                pending = self.pending_lease()
                can_start = changed and not pending and not self.anonymous_running
                if can_start:
                    self.anonymous_running = True
            previous = current
            if can_start:
                reason = "anonymous CUDA process set changed"
                threading.Thread(target=self.anonymous_rebalance, args=(reason,), daemon=True).start()

    def lease_reaper(self) -> None:
        while not self.stopping.wait(5.0):
            now = time.time()
            with self.cv:
                expired = [lease.token for lease in self.leases.values()
                           if lease.ttl > 0 and now - lease.heartbeat_at > lease.ttl]
            for token in expired:
                try:
                    self.release(token, "expired lease")
                except Exception as exc:
                    LOG.error("failed to expire lease %s: %s", token, exc)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    broker: Broker

    def _handle(self) -> None:
        if self.command == "GET" and self.path.split("?", 1)[0] == "/.well-known/ollama-unify-gpu-negotiator":
            body = json.dumps(discovery_document(), indent=2, sort_keys=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            self.broker.proxy_enter()
        except TimeoutError as exc:
            self.send_error(503, str(exc))
            return
        response_started = False
        backend = None
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            body = clamp_request(self.path, self.headers.get("Content-Type", ""), body)
            headers = {key: value for key, value in self.headers.items()
                       if key.lower() not in HOP_HEADERS and key.lower() not in {"host", "content-length"}}
            backend = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=None)
            backend.request(self.command, self.path, body=body if body else None, headers=headers)
            response = backend.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS and key.lower() != "content-length":
                    self.send_header(key, value)
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                self.send_header("Content-Length", content_length)
            else:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            response_started = True
            reader = getattr(response, "read1", response.read)
            while True:
                chunk = reader(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            LOG.error("proxy error %s %s: %s", self.command, self.path, exc)
            if not response_started:
                self.send_error(502, f"Ollama backend unavailable: {exc}")
        finally:
            if backend is not None:
                backend.close()
            self.broker.proxy_exit()

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_HEAD = _handle
    do_OPTIONS = _handle

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("proxy %s - %s", self.address_string(), fmt % args)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ControlHandler(socketserver.StreamRequestHandler):
    broker: Broker

    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline(1024 * 1024))
            action = request.get("action")
            if action == "acquire":
                result = self.broker.acquire(
                    str(request.get("owner") or "unknown"),
                    max(0, int(request.get("requested_mib") or 0)),
                    max(0, int(request.get("ttl", DEFAULT_LEASE_TTL))),
                )
            elif action == "ready":
                result = self.broker.ready(str(request.get("token") or ""))
            elif action == "prepare":
                result = self.broker.prepare(str(request.get("token") or ""))
            elif action == "release":
                result = self.broker.release(str(request.get("token") or ""))
            elif action == "heartbeat":
                result = self.broker.heartbeat(str(request.get("token") or ""))
            elif action == "status":
                result = self.broker.status()
            else:
                raise ValueError(f"unknown action: {action}")
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        self.wfile.write(json.dumps(result, separators=(",", ":")).encode() + b"\n")


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def send_control(payload: dict[str, Any]) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(DRAIN_TIMEOUT + UNLOAD_TIMEOUT + ANON_MAX_DRAIN + 10)
        client.connect(CONTROL_SOCKET)
        client.sendall(json.dumps(payload).encode() + b"\n")
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        response = json.loads(b"".join(chunks) or b"{}")
    finally:
        client.close()
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "negotiator request failed"))
    return response


def serve() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    broker = Broker()
    control_path = Path(CONTROL_SOCKET)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    if control_path.exists() or control_path.is_socket():
        control_path.unlink()

    ControlHandler.broker = broker
    ProxyHandler.broker = broker
    control = ThreadingUnixServer(CONTROL_SOCKET, ControlHandler)
    os.chmod(CONTROL_SOCKET, 0o660)
    proxy = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)

    def shutdown(_signum: int, _frame: Any) -> None:
        broker.stopping.set()
        threading.Thread(target=control.shutdown, daemon=True).start()
        threading.Thread(target=proxy.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=control.serve_forever, daemon=True).start()
    threading.Thread(target=broker.anonymous_watcher, daemon=True).start()
    threading.Thread(target=broker.lease_reaper, daemon=True).start()
    LOG.info("proxy listening on %s:%s; backend %s:%s; control %s",
             LISTEN_HOST, LISTEN_PORT, BACKEND_HOST, BACKEND_PORT, CONTROL_SOCKET)
    try:
        proxy.serve_forever()
    finally:
        broker.stopping.set()
        control.shutdown()
        control.server_close()
        proxy.server_close()
        try:
            control_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def lease_run(args: argparse.Namespace) -> int:
    if not args.ready_command:
        raise RuntimeError("run requires --ready-command so Ollama cannot reload before external CUDA allocation")
    owner = args.owner or f"{os.environ.get('USER', 'user')}:{os.getpid()}"
    acquired = send_control({"action": "acquire", "owner": owner,
                             "requested_mib": args.vram_mib, "ttl": args.ttl})
    token = acquired["lease"]["token"]
    env = os.environ.copy()
    env["OLLAMA_UNIFY_GPU_LEASE"] = token
    child = None
    stop_heartbeat = threading.Event()
    try:
        child = subprocess.Popen(args.command, env=env)
        deadline = time.monotonic() + args.ready_timeout
        while child.poll() is None and time.monotonic() < deadline:
            ready = subprocess.run(args.ready_command, shell=True, env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ready.returncode == 0:
                break
            time.sleep(1.0)
        if child.poll() is not None:
            raise RuntimeError("external command exited before its readiness check passed")
        if time.monotonic() >= deadline:
            raise TimeoutError("external workload did not become ready before timeout")
        send_control({"action": "ready", "token": token})

        def heartbeat() -> None:
            interval = max(2.0, min(30.0, args.ttl / 3 if args.ttl else 30.0))
            while not stop_heartbeat.wait(interval):
                try:
                    send_control({"action": "heartbeat", "token": token})
                except Exception as exc:
                    print(f"ollama-unify lease heartbeat failed: {exc}", file=sys.stderr)

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            return child.wait()
        except KeyboardInterrupt:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            return 130
    finally:
        stop_heartbeat.set()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        try:
            send_control({"action": "release", "token": token})
        except Exception as exc:
            print(f"ollama-unify lease release failed: {exc}", file=sys.stderr)


def self_test() -> int:
    global MAX_CONTEXT
    MAX_CONTEXT = 8192
    original = json.dumps({"model": "test", "options": {
        "num_gpu": 999, "main_gpu": 2, "num_ctx": 262144,
    }}).encode()
    clamped = json.loads(clamp_request("/api/generate", "application/json", original))
    assert clamped["options"]["num_gpu"] == -1
    assert "main_gpu" not in clamped["options"]
    assert clamped["options"]["num_ctx"] == 8192
    assert split_address("0.0.0.0") == ("0.0.0.0", 11434)
    assert split_address("127.0.0.1:11435") == ("127.0.0.1", 11435)
    document = discovery_document()
    assert document["schema"] == "io.ollama-unify.gpu-negotiator.discovery.v1"
    assert document["commands"]["discover"] == "docker gpu discover"
    assert "num_gpu" in agent_instructions_text()
    print("negotiator self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic Ollama/external CUDA lease negotiator")
    sub = parser.add_subparsers(dest="command_name", required=True)
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("self-test")
    sub.add_parser("discover")
    sub.add_parser("agent-instructions")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--owner", default="")
    acquire.add_argument("--vram-mib", type=int, default=0)
    acquire.add_argument("--ttl", type=int, default=DEFAULT_LEASE_TTL)
    acquire.add_argument("--token-only", action="store_true")
    for name in ("ready", "prepare", "release", "heartbeat"):
        command = sub.add_parser(name)
        command.add_argument("token")
    run = sub.add_parser("run")
    run.add_argument("--owner", default="")
    run.add_argument("--vram-mib", type=int, default=0)
    run.add_argument("--ttl", type=int, default=DEFAULT_LEASE_TTL)
    run.add_argument("--ready-command", required=True)
    run.add_argument("--ready-timeout", type=float, default=300.0)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command_name == "serve":
        return serve()
    if args.command_name == "self-test":
        return self_test()
    if args.command_name == "discover":
        print(json.dumps(discovery_document(), indent=2, sort_keys=True))
        return 0
    if args.command_name == "agent-instructions":
        print(agent_instructions_text(), end="")
        return 0
    if args.command_name == "status":
        result = send_control({"action": "status"})
    elif args.command_name == "acquire":
        result = send_control({"action": "acquire", "owner": args.owner,
                               "requested_mib": args.vram_mib, "ttl": args.ttl})
    elif args.command_name in ("ready", "prepare", "release", "heartbeat"):
        result = send_control({"action": args.command_name, "token": args.token})
    elif args.command_name == "run":
        if not args.command:
            parser.error("run requires a command after --")
        if args.command[0] == "--":
            args.command = args.command[1:]
        return lease_run(args)
    else:
        parser.error("unknown command")
    if args.command_name == "acquire" and args.token_only:
        print(result["lease"]["token"])
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"ollama-unify negotiator: {exc}", file=sys.stderr)
        raise SystemExit(1)
NEGOTIATOR
}

render_gpu_negotiator_cli() {
  cat <<'LEASE_CLI'
#!/bin/sh
# Generated by ollama-unify.
exec /usr/local/libexec/ollama-unify-gpu-negotiator "$@"
LEASE_CLI
}

render_docker_gpu_lease_plugin() {
  cat <<'DOCKER_PLUGIN'
#!/bin/sh
# Generated by ollama-unify. Docker CLI plugin entrypoint.
if [ "${1:-}" = "docker-cli-plugin-metadata" ]; then
  printf '%s\n' '{"SchemaVersion":"0.1.0","Vendor":"ollama-unify","Version":"1.0.0","ShortDescription":"Negotiate CUDA VRAM with Ollama","URL":"https://github.com/robit-man/ollama-unify"}'
  exit 0
fi
if [ "${1:-}" = "gpu" ]; then shift; fi
exec "${OLLAMA_UNIFY_GPU_LEASE_CLI:-/usr/local/bin/ollama-unify-gpu-lease}" "$@"
DOCKER_PLUGIN
}

render_gpu_agent_instructions() {
  cat <<'AGENT_INSTRUCTIONS'
# Host CUDA negotiation

This host runs the ollama-unify GPU lease broker. Before creating, starting, or resizing any Docker/container/service deployment that uses CUDA:

1. Run `docker gpu discover` and inspect the selected GPUs and current policy.
2. Launch long-running CUDA services with `docker gpu run --owner NAME --vram-mib MIB --ready-command 'CHECK' -- COMMAND`. The readiness check must pass only after CUDA models are resident.
3. For an independently supervised service, use `acquire`, load the CUDA models, call `ready`, call `prepare` before any VRAM growth, and call `release` only after freeing CUDA memory.
4. Never bypass the broker by assuming free VRAM from a static scan. Anonymous allocation is reactive and cannot prevent the first CUDA OOM.
5. Ollama API `num_gpu` counts offloaded layers, not physical GPUs. Keep it automatic (`-1`); every selected GPU remains available to the scheduler.

Machine-readable discovery: `/usr/local/share/ollama-unify/gpu-negotiator.json` or `http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator`.
AGENT_INSTRUCTIONS
}

render_global_codex_gpu_block() {
  cat <<'CODEX_BLOCK'
<!-- BEGIN ollama-unify GPU negotiator -->
## CUDA deployment policy

Before changing or starting any Docker/container/service workload that uses CUDA on this host, run `docker gpu discover`. Use `docker gpu run` for foreground deployments or the acquire → ready → prepare/resize → release protocol for independently supervised services. The external workload must allocate first so Ollama can refit from remaining live VRAM. Do not rely on static free-VRAM scans or bypass the broker; anonymous CUDA allocation is only best-effort. Full instructions are at `/usr/local/share/ollama-unify/AGENTS.md`.
<!-- END ollama-unify GPU negotiator -->
CODEX_BLOCK
}

detect_ollama_proxy_listen() {
  local listen="${OLLAMA_SAFE_NEGOTIATOR_LISTEN:-}" effective_env
  if [ -z "$listen" ] && [ -r "$SAFETY_NEGOTIATOR_CONFIG_PATH" ]; then
    listen=$(awk -F= '$1 == "OLLAMA_UNIFY_LISTEN" { sub(/^[^=]*=/, ""); gsub(/^"|"$/, ""); print; exit }' \
      "$SAFETY_NEGOTIATOR_CONFIG_PATH" 2>/dev/null || true)
  fi
  if [ -z "$listen" ]; then
    effective_env=$(systemctl show ollama.service -p Environment --value 2>/dev/null || true)
    listen=$(printf '%s\n' "$effective_env" | grep -o 'OLLAMA_HOST=[^ "[:space:]]*' \
      | tail -n 1 | cut -d= -f2- || true)
  fi
  listen="${listen#http://}"
  listen="${listen#https://}"
  listen="${listen%/}"
  [ -n "$listen" ] || listen="127.0.0.1"
  if [ "$listen" = "$SAFETY_OLLAMA_BACKEND" ]; then listen="127.0.0.1:11434"; fi
  [[ "$listen" == *:* ]] || listen="${listen}:11434"
  [[ "$listen" =~ ^[A-Za-z0-9_.:-]+$ ]] \
    || { err "invalid negotiator listen address: $listen"; exit 2; }
  printf '%s' "$listen"
}

install_gpu_negotiator() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  command -v python3 >/dev/null 2>&1 \
    || { err "python3 is required for the streaming GPU negotiator"; exit 2; }

  local proxy_listen service_user service_group access_group unit_dir config_dir helper_dir cli_dir
  local plugin_dir selected_ids
  local drain_timeout unload_timeout lease_ttl anon_poll anon_settle anon_max_drain
  proxy_listen=$(detect_ollama_proxy_listen)
  service_user=$(systemctl show ollama.service -p User --value 2>/dev/null || true)
  service_group=$(systemctl show ollama.service -p Group --value 2>/dev/null || true)
  [ -n "$service_user" ] || service_user="ollama"
  [ -n "$service_group" ] || service_group="$service_user"
  access_group="${OLLAMA_SAFE_NEGOTIATOR_GROUP:-$service_group}"
  if [ "$EUID" -ne 0 ]; then access_group="${OLLAMA_SAFE_NEGOTIATOR_GROUP:-$(id -gn)}"; fi
  [[ "$service_user" =~ ^[A-Za-z0-9_.@-]+$ ]] \
    || { err "cannot install negotiator for unsafe service user value: $service_user"; exit 2; }
  [[ "$service_group" =~ ^[A-Za-z0-9_.@-]+$ ]] \
    || { err "cannot install negotiator for unsafe service group value: $service_group"; exit 2; }
  [[ "$access_group" =~ ^[A-Za-z0-9_.@-]+$ ]] \
    || { err "cannot install negotiator for unsafe access group value: $access_group"; exit 2; }

  drain_timeout="${OLLAMA_SAFE_NEGOTIATOR_DRAIN_TIMEOUT:-300}"
  unload_timeout="${OLLAMA_SAFE_NEGOTIATOR_UNLOAD_TIMEOUT:-120}"
  lease_ttl="${OLLAMA_SAFE_NEGOTIATOR_LEASE_TTL:-300}"
  anon_poll="${OLLAMA_SAFE_NEGOTIATOR_ANON_POLL:-0.5}"
  anon_settle="${OLLAMA_SAFE_NEGOTIATOR_ANON_SETTLE:-2}"
  anon_max_drain="${OLLAMA_SAFE_NEGOTIATOR_ANON_MAX_DRAIN:-15}"
  [[ "$drain_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_DRAIN_TIMEOUT must be numeric"; exit 2; }
  [[ "$unload_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_UNLOAD_TIMEOUT must be numeric"; exit 2; }
  [[ "$lease_ttl" =~ ^[0-9]+$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_LEASE_TTL must be an unsigned integer"; exit 2; }
  [[ "$anon_poll" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_ANON_POLL must be numeric"; exit 2; }
  [[ "$anon_settle" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_ANON_SETTLE must be numeric"; exit 2; }
  [[ "$anon_max_drain" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_ANON_MAX_DRAIN must be numeric"; exit 2; }

  unit_dir="${SAFETY_NEGOTIATOR_UNIT_PATH%/*}"
  config_dir="${SAFETY_NEGOTIATOR_CONFIG_PATH%/*}"
  helper_dir="${SAFETY_NEGOTIATOR_PATH%/*}"
  cli_dir="${SAFETY_NEGOTIATOR_CLI_PATH%/*}"
  plugin_dir="${SAFETY_DOCKER_PLUGIN_PATH%/*}"
  selected_ids=$(csv_from_array "${SAFETY_DEVICE_IDS[@]}")
  "${elevate[@]}" mkdir -p "$unit_dir" "$config_dir" "$helper_dir" "$cli_dir" \
    "$plugin_dir" "$SAFETY_DISCOVERY_DIR"

  render_gpu_negotiator_script | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_PATH" >/dev/null
  render_gpu_negotiator_cli | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_CLI_PATH" >/dev/null
  render_docker_gpu_lease_plugin | "${elevate[@]}" tee "$SAFETY_DOCKER_PLUGIN_PATH" >/dev/null
  if "${elevate[@]}" test -f "$SAFETY_LEGACY_DOCKER_PLUGIN_PATH" \
    && "${elevate[@]}" grep -q '^# Generated by ollama-unify' "$SAFETY_LEGACY_DOCKER_PLUGIN_PATH"; then
    "${elevate[@]}" rm -f "$SAFETY_LEGACY_DOCKER_PLUGIN_PATH"
  fi
  render_gpu_agent_instructions | "${elevate[@]}" tee "$SAFETY_AGENT_INSTRUCTIONS_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_NEGOTIATOR_PATH" "$SAFETY_NEGOTIATOR_CLI_PATH" \
    "$SAFETY_DOCKER_PLUGIN_PATH"
  "${elevate[@]}" chmod 0644 "$SAFETY_AGENT_INSTRUCTIONS_PATH"

  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf 'OLLAMA_UNIFY_BACKEND="%s"\n' "$SAFETY_OLLAMA_BACKEND"
    printf 'OLLAMA_UNIFY_BACKEND_TYPE="%s"\n' "$SAFETY_BACKEND"
    printf 'OLLAMA_UNIFY_SELECTED_GPUS="%s"\n' "$selected_ids"
    printf 'OLLAMA_UNIFY_LISTEN="%s"\n' "$proxy_listen"
    printf 'OLLAMA_UNIFY_SOCKET="%s"\n' "$SAFETY_NEGOTIATOR_SOCKET"
    printf 'OLLAMA_UNIFY_MAX_CONTEXT="%s"\n' "$SAFETY_CONTEXT_LENGTH"
    printf 'OLLAMA_UNIFY_DRAIN_TIMEOUT="%s"\n' "$drain_timeout"
    printf 'OLLAMA_UNIFY_UNLOAD_TIMEOUT="%s"\n' "$unload_timeout"
    printf 'OLLAMA_UNIFY_LEASE_TTL="%s"\n' "$lease_ttl"
    printf 'OLLAMA_UNIFY_ANON_POLL="%s"\n' "$anon_poll"
    printf 'OLLAMA_UNIFY_ANON_SETTLE="%s"\n' "$anon_settle"
    printf 'OLLAMA_UNIFY_ANON_MAX_DRAIN="%s"\n' "$anon_max_drain"
  } | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_CONFIG_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_NEGOTIATOR_CONFIG_PATH"
  "${elevate[@]}" "$SAFETY_NEGOTIATOR_PATH" discover \
    | "${elevate[@]}" tee "$SAFETY_DISCOVERY_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_DISCOVERY_PATH"

  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    cat <<UNIT
[Unit]
Description=Ollama dynamic GPU lease negotiator and API proxy
Documentation=https://github.com/robit-man/ollama-unify
Wants=ollama.service network-online.target
After=ollama.service network-online.target
StartLimitIntervalSec=5min
StartLimitBurst=5

[Service]
Type=simple
User=$service_user
Group=$access_group
EnvironmentFile=$SAFETY_NEGOTIATOR_CONFIG_PATH
RuntimeDirectory=ollama-unify
RuntimeDirectoryMode=0750
ExecStartPre=$SAFETY_NEGOTIATOR_PATH self-test
ExecStart=$SAFETY_NEGOTIATOR_PATH serve
Restart=on-failure
RestartSec=5s
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
LockPersonality=yes
UMask=0007

[Install]
WantedBy=multi-user.target
UNIT
  } | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_UNIT_PATH" >/dev/null

  "${elevate[@]}" "$SAFETY_NEGOTIATOR_PATH" self-test >/dev/null
  ok "dynamic GPU negotiator installed: $SAFETY_NEGOTIATOR_PATH"
  ok "GPU lease client installed: $SAFETY_NEGOTIATOR_CLI_PATH"
  ok "Docker discovery plugin installed: $SAFETY_DOCKER_PLUGIN_PATH"
  ok "agent discovery manifest installed: $SAFETY_DISCOVERY_PATH"
  say "  Ollama backend: $SAFETY_OLLAMA_BACKEND; negotiated API: $proxy_listen"
}

install_global_codex_gpu_instructions() {
  local sudo_pfx="$1" enabled="${OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY:-1}"
  [ "$enabled" = 1 ] || { [ "$enabled" = 0 ] && return; err "OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY must be 0 or 1"; exit 2; }

  local agent_user agent_group agent_home agent_dir agent_file temp_file
  agent_user="${SUDO_USER:-${USER:-}}"
  [ -n "$agent_user" ] || return
  agent_home=$(getent passwd "$agent_user" 2>/dev/null | awk -F: 'NR == 1 { print $6 }')
  [ -n "$agent_home" ] || return
  agent_dir="$agent_home/.codex"
  [ -d "$agent_dir" ] || return
  agent_file="$agent_dir/AGENTS.md"
  if [ -L "$agent_file" ]; then
    warn "skipping global Codex discovery because $agent_file is a symlink"
    return
  fi
  agent_group=$(id -gn "$agent_user" 2>/dev/null || printf '%s' "$agent_user")
  temp_file=$(mktemp)
  if [ -f "$agent_file" ]; then
    awk '
      $0 == "<!-- BEGIN ollama-unify GPU negotiator -->" { skip=1; next }
      $0 == "<!-- END ollama-unify GPU negotiator -->" { skip=0; next }
      !skip { print }
    ' "$agent_file" > "$temp_file"
  fi
  if [ -s "$temp_file" ]; then printf '\n' >> "$temp_file"; fi
  render_global_codex_gpu_block >> "$temp_file"

  if [ "$EUID" -eq 0 ] || [ "$agent_user" != "${USER:-}" ]; then
    local -a elevate=()
    [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
    "${elevate[@]}" install -o "$agent_user" -g "$agent_group" -m 0644 "$temp_file" "$agent_file"
  else
    install -m 0644 "$temp_file" "$agent_file"
  fi
  rm -f "$temp_file"
  ok "global Codex CUDA discovery installed: $agent_file"
}

render_safety_service_directives() {
  if [ "$SYSTEMD_VERSION" -ge 235 ]; then
    printf '%s\n' "UnsetEnvironment=OLLAMA_LLM_LIBRARY"
    if [ "$SAFETY_VRAM_RESERVE_MIB" -eq 0 ]; then
      printf '%s\n' "UnsetEnvironment=OLLAMA_GPU_OVERHEAD"
    fi
    if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
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
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    # The negotiator owns the public Ollama port and serializes model loads
    # around external GPU leases. Ollama itself is reachable only through the
    # loopback backend, so cooperative draining cannot be bypassed remotely.
    printf '%s\n' "Environment=\"OLLAMA_HOST=${SAFETY_OLLAMA_BACKEND}\""
  fi
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

  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    command -v python3 >/dev/null 2>&1 \
      || { err "python3 is required before Ollama can be routed through the GPU negotiator"; exit 2; }
  fi

  "${elevate[@]}" mkdir -p "$dropin_dir" "$preflight_dir"
  render_safety_preflight_script | "${elevate[@]}" tee "$SAFETY_PREFLIGHT_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_PREFLIGHT_PATH"
  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf '[Unit]\nStartLimitIntervalSec=5min\nStartLimitBurst=2\n\n[Service]\n'
    render_safety_service_directives
  } | "${elevate[@]}" tee "$safety_file" >/dev/null

  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    install_gpu_negotiator "$sudo_pfx"
    install_global_codex_gpu_instructions "$sudo_pfx"
  fi

  "${elevate[@]}" systemctl daemon-reload
  "${elevate[@]}" systemctl enable ollama.service >/dev/null
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    "${elevate[@]}" systemctl enable ollama-unify-negotiator.service >/dev/null
  fi
  if [ "$SYSTEMD_VERSION" -ge 247 ]; then
    if "${elevate[@]}" systemctl enable --now systemd-oomd.service >/dev/null 2>&1; then
      ok "systemd-oomd is enabled for proactive memory-pressure kills"
    else
      warn "systemd-oomd could not be enabled; MemoryMax remains active, but PSI-based killing is unavailable"
    fi
  fi
  if command -v systemd-analyze >/dev/null 2>&1; then
    if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
      "${elevate[@]}" systemd-analyze verify ollama.service ollama-unify-negotiator.service
    else
      "${elevate[@]}" systemd-analyze verify ollama.service
    fi
  fi
  ok "memory-pressure preflight installed: $SAFETY_PREFLIGHT_PATH"
  ok "late-priority safety drop-in installed: $safety_file"
  install_reconcile_watchdog "$sudo_pfx"
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

  local was_active=0 negotiator_was_active=0
  if systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1; then
    negotiator_was_active=1
    $SUDO systemctl stop ollama-unify-negotiator.service
    ok "ollama-unify-negotiator.service stopped while its policy is replaced"
  fi
  if systemctl is-active ollama.service >/dev/null 2>&1; then
    was_active=1
    $SUDO systemctl stop ollama.service
    ok "ollama.service stopped while its policy is replaced"
  fi

  hdr "Installing Ollama safety policy"
  install_systemd_safety_policy "$SUDO"

  if [ "$was_active" = 1 ]; then
    $SUDO systemctl start ollama.service || true
    if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
      $SUDO systemctl start ollama-unify-negotiator.service || true
    fi
    if systemctl is-active ollama.service >/dev/null 2>&1; then
      ok "ollama.service restarted under the new policy"
    else
      warn "ollama.service remains stopped because the safety condition refused the restart"
    fi
  else
    say "  ollama.service was already stopped and has been left stopped."
    if [ "$negotiator_was_active" = 1 ]; then
      warn "the negotiator was active without Ollama and has been left stopped with it"
    fi
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
# ──────────────────────────────── Ollama update reconciliation
# An Ollama upgrade rewrites /etc/systemd/system/ollama.service and restarts the
# daemon. The unit it installs carries no OLLAMA_HOST, so the pinned loopback
# backend exists only in our late-priority drop-in. If that drop-in is lost or
# drifts, Ollama falls back to its built-in 0.0.0.0:11434 and collides head-on
# with the negotiator that already owns that address. These helpers detect the
# drift, repair it, and drive upgrades inside a safe stop → repin → start
# envelope so the proxy never races the daemon it fronts.

DRIFT_FINDINGS=()

ollama_binary_path() {
  local exec_start path
  exec_start=$(systemctl show ollama.service -p ExecStart --value 2>/dev/null || true)
  path=$(printf '%s' "$exec_start" | grep -oE 'path=[^ ;]+' | head -n1 | cut -d= -f2- || true)
  if [ -n "$path" ] && [ -x "$path" ]; then printf '%s' "$path"; return 0; fi
  command -v ollama 2>/dev/null || return 1
}

ollama_installed_version() {
  local bin ver
  bin=$(ollama_binary_path) || return 1
  ver=$("$bin" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)
  [ -n "$ver" ] || return 1
  printf '%s' "$ver"
}

ollama_latest_version() {
  [ "$HAS_CURL" = 1 ] || return 1
  curl -fsSL --max-time 10 "$SAFETY_OLLAMA_RELEASE_API" 2>/dev/null \
    | grep -oE '"tag_name"[[:space:]]*:[[:space:]]*"[^"]+"' | head -n1 \
    | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1
}

# true when $1 sorts strictly before $2 under version ordering
version_lt() {
  [ "$1" = "$2" ] && return 1
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]
}

port_holder_pid() {
  local port="$1"
  command -v ss >/dev/null 2>&1 || return 1
  ss -tlnpH 2>/dev/null | awk -v p=":${port}\$" '$4 ~ p {print; exit}' \
    | grep -oE 'pid=[0-9]+' | head -n1 | cut -d= -f2
}

effective_ollama_host() {
  systemctl show ollama.service -p Environment --value 2>/dev/null \
    | tr ' ' '\n' | grep -E '^OLLAMA_HOST=' | tail -n1 | cut -d= -f2- | tr -d '"'
}

negotiator_config_value() {
  local key="$1"
  [ -r "$SAFETY_NEGOTIATOR_CONFIG_PATH" ] || return 1
  awk -F= -v k="$key" '$1 == k { sub(/^[^=]*=/, ""); gsub(/^"|"$/, ""); print; exit }' \
    "$SAFETY_NEGOTIATOR_CONFIG_PATH"
}

state_value() {
  local key="$1"
  [ -r "$SAFETY_STATE_PATH" ] || return 1
  awk -F= -v k="$key" '$1 == k { sub(/^[^=]*=/, ""); gsub(/^"|"$/, ""); print; exit }' \
    "$SAFETY_STATE_PATH"
}

render_reconcile_state() {
  local version
  version=$(ollama_installed_version || printf 'unknown')
  printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'OLLAMA_UNIFY_PINNED_BACKEND="%s"\n' "$SAFETY_OLLAMA_BACKEND"
  printf 'OLLAMA_UNIFY_PINNED_LISTEN="%s"\n' "$(detect_ollama_proxy_listen)"
  printf 'OLLAMA_UNIFY_OLLAMA_VERSION="%s"\n' "$version"
  printf 'OLLAMA_UNIFY_OLLAMA_BINARY="%s"\n' "$(ollama_binary_path || printf '')"
}

# Populates DRIFT_FINDINGS with everything that would break the proxy↔backend pair.
collect_policy_drift() {
  DRIFT_FINDINGS=()
  local dropin="/etc/systemd/system/ollama.service.d/zzz-ollama-unify-safety.conf"
  local host backend listen backend_port holder_pid main_pid holder_comm

  [ -r "$dropin" ] || DRIFT_FINDINGS+=(
    "safety drop-in is missing ($dropin); Ollama would fall back to its built-in 0.0.0.0:11434")

  host=$(effective_ollama_host || true)
  if [ -z "$host" ]; then
    DRIFT_FINDINGS+=("ollama.service exposes no OLLAMA_HOST; the daemon would bind its default address")
  elif [ "$host" != "$SAFETY_OLLAMA_BACKEND" ]; then
    DRIFT_FINDINGS+=("ollama.service OLLAMA_HOST is $host; the pinned backend is $SAFETY_OLLAMA_BACKEND")
  fi

  if [ -r "$SAFETY_NEGOTIATOR_CONFIG_PATH" ]; then
    backend=$(negotiator_config_value OLLAMA_UNIFY_BACKEND || true)
    listen=$(negotiator_config_value OLLAMA_UNIFY_LISTEN || true)
    if [ -n "$backend" ] && [ "$backend" != "$SAFETY_OLLAMA_BACKEND" ]; then
      DRIFT_FINDINGS+=("negotiator OLLAMA_UNIFY_BACKEND is $backend; the pinned backend is $SAFETY_OLLAMA_BACKEND")
    fi
    if [ -n "$listen" ] && [ "$listen" = "$SAFETY_OLLAMA_BACKEND" ]; then
      DRIFT_FINDINGS+=("negotiator listen address equals the backend address ($listen); the proxy would loop onto itself")
    fi
  fi

  backend_port="${SAFETY_OLLAMA_BACKEND##*:}"
  holder_pid=$(port_holder_pid "$backend_port" || true)
  main_pid=$(systemctl show ollama.service -p MainPID --value 2>/dev/null || printf '0')
  if [ -n "$holder_pid" ] && [ "$holder_pid" != "${main_pid:-0}" ]; then
    holder_comm=$(ps -o comm= -p "$holder_pid" 2>/dev/null || printf 'unknown')
    DRIFT_FINDINGS+=("backend port $backend_port is held by PID $holder_pid ($holder_comm), not by ollama.service")
  fi
}

# Returns 0 when the negotiated stack answers end to end.
verify_negotiated_stack() {
  local listen probe host port
  systemctl is-active ollama.service >/dev/null 2>&1 \
    || { warn "ollama.service is not active"; return 1; }
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1 \
      || { warn "ollama-unify-negotiator.service is not active"; return 1; }
  fi
  listen=$(negotiator_config_value OLLAMA_UNIFY_LISTEN 2>/dev/null || printf '')
  [ -n "$listen" ] || listen="$SAFETY_OLLAMA_BACKEND"
  host="${listen%:*}"; port="${listen##*:}"
  [ "$host" = "0.0.0.0" ] && host="127.0.0.1"
  [ "$HAS_CURL" = 1 ] || { warn "curl is unavailable; skipping the end-to-end probe"; return 0; }
  probe=$(curl -fsS --max-time 15 "http://${host}:${port}/api/tags" 2>/dev/null || true)
  [ -n "$probe" ] || { warn "the negotiated API at ${host}:${port} did not answer /api/tags"; return 1; }
  ok "negotiated API answers on ${host}:${port}"
}

# Ordered cycle: the proxy must release its socket before the backend moves.
cycle_negotiated_stack() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  "${elevate[@]}" systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
  "${elevate[@]}" systemctl restart ollama.service || {
    err "ollama.service refused to start; the safety condition or the backend port is still blocked"
    return 1
  }
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    "${elevate[@]}" systemctl start ollama-unify-negotiator.service || {
      err "ollama-unify-negotiator.service refused to start"
      return 1
    }
  fi
}

report_update_status() {
  local installed latest recorded
  installed=$(ollama_installed_version || printf '')
  recorded=$(state_value OLLAMA_UNIFY_OLLAMA_VERSION 2>/dev/null || printf '')
  hdr "Ollama release status"
  if [ -n "$installed" ]; then
    say "  Installed: $installed ($(ollama_binary_path || printf 'binary not found'))"
  else
    warn "  Installed: could not read a version from the Ollama binary"
  fi
  if [ -n "$recorded" ] && [ -n "$installed" ] && [ "$recorded" != "$installed" ]; then
    warn "  Ollama moved $recorded → $installed since the policy was last applied"
  fi
  latest=$(ollama_latest_version || printf '')
  if [ -z "$latest" ]; then
    say "  Latest:    unavailable (no network or GitHub API unreachable)"
  elif [ -z "$installed" ]; then
    say "  Latest:    $latest"
  elif version_lt "$installed" "$latest"; then
    warn "  Latest:    $latest — an update is available; run --update-ollama to take it safely"
  else
    ok "  Latest:    $latest — Ollama is current"
  fi
}

report_policy_drift() {
  collect_policy_drift
  hdr "Pinned topology"
  say "  Ollama backend:  $SAFETY_OLLAMA_BACKEND"
  say "  Negotiated API:  $(negotiator_config_value OLLAMA_UNIFY_LISTEN 2>/dev/null || printf 'not installed')"
  if [ ${#DRIFT_FINDINGS[@]} -eq 0 ]; then
    ok "no policy drift; the proxy and the backend agree on their addresses"
    return 0
  fi
  hdr "Policy drift"
  local finding
  for finding in "${DRIFT_FINDINGS[@]}"; do warn "  $finding"; done
  return 1
}

check_ollama_update() {
  banner
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--check-update requires a systemd host"; exit 2; }
  build_safety_profile
  report_update_status
  if report_policy_drift; then
    say ""
    ok "nothing to reconcile"
  else
    say ""
    warn "run --reconcile to repin Ollama and restart the pair in the correct order"
  fi
}

reconcile_after_update() {
  banner
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--reconcile requires a systemd host"; exit 2; }
  systemctl cat ollama.service >/dev/null 2>&1 \
    || { err "ollama.service was not found"; exit 2; }
  build_safety_profile
  report_update_status

  local drifted=0
  report_policy_drift || drifted=1

  local installed recorded
  installed=$(ollama_installed_version || printf '')
  recorded=$(state_value OLLAMA_UNIFY_OLLAMA_VERSION 2>/dev/null || printf '')
  [ -n "$installed" ] && [ -n "$recorded" ] && [ "$installed" != "$recorded" ] && drifted=1

  if [ "$drifted" = 0 ]; then
    say ""
    ok "policy already matches the running stack; nothing was changed"
    verify_negotiated_stack || true
    return 0
  fi

  local SUDO=""
  if [ "$EUID" -ne 0 ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo is required to reconcile the systemd policy"; exit 2; }
    SUDO="sudo"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  hdr "Reapplying the pinned safety policy"
  $SUDO systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
  $SUDO systemctl stop ollama.service 2>/dev/null || true
  install_systemd_safety_policy "$SUDO"

  hdr "Restarting the negotiated pair"
  cycle_negotiated_stack "$SUDO" || exit 1
  verify_negotiated_stack || exit 1
  ok "Ollama is repinned to $SAFETY_OLLAMA_BACKEND behind the negotiator"
}

update_ollama() {
  banner
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--update-ollama requires a systemd host"; exit 2; }
  [ "$HAS_CURL" = 1 ] || { err "curl is required to download an Ollama update"; exit 2; }
  build_safety_profile
  report_update_status

  local installed latest
  installed=$(ollama_installed_version || printf '')
  latest=$(ollama_latest_version || printf '')
  if [ -z "$latest" ]; then
    err "the latest Ollama release could not be determined; refusing to run the installer blind"
    exit 2
  fi
  if [ -n "$installed" ] && ! version_lt "$installed" "$latest"; then
    say ""
    ok "Ollama $installed is already current; nothing to update"
    exit 0
  fi

  say ""
  say "  The official installer rewrites /etc/systemd/system/ollama.service and restarts"
  say "  the daemon. ollama-unify will stop the negotiator first, let the installer run,"
  say "  then repin the backend to $SAFETY_OLLAMA_BACKEND before the proxy comes back."
  if [ -t 0 ] && [ "${OLLAMA_UNIFY_ASSUME_YES:-0}" != "1" ]; then
    confirm "Update Ollama ${installed:-unknown} → $latest now?" "Y" \
      || { say "  Left unchanged."; exit 0; }
  fi

  local SUDO=""
  if [ "$EUID" -ne 0 ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo is required to update Ollama"; exit 2; }
    SUDO="sudo"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  hdr "Quiescing the negotiated pair"
  $SUDO systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
  ok "negotiator stopped; the public address is free while the installer runs"
  $SUDO systemctl stop ollama.service 2>/dev/null || true
  ok "ollama.service stopped"

  hdr "Running the official Ollama installer"
  if ! curl -fsSL "$SAFETY_OLLAMA_INSTALL_URL" | $SUDO sh; then
    err "the Ollama installer failed; reconciling the previous policy before exiting"
    install_systemd_safety_policy "$SUDO"
    cycle_negotiated_stack "$SUDO" || true
    exit 1
  fi

  hdr "Repinning Ollama behind the negotiator"
  install_systemd_safety_policy "$SUDO"
  cycle_negotiated_stack "$SUDO" || exit 1
  verify_negotiated_stack || exit 1
  ok "Ollama updated to $(ollama_installed_version || printf "$latest") and repinned to $SAFETY_OLLAMA_BACKEND"
}

# Standalone repair helper: no repo checkout, no python, reads the installed state.
render_reconcile_helper() {
  cat <<'RECONCILE'
#!/usr/bin/env bash
# Managed by ollama-unify — repins Ollama after an out-of-band upgrade.
#
# The Ollama installer rewrites ollama.service and restarts the daemon. The
# loopback backend address lives only in the ollama-unify drop-in, so an upgrade
# that clears or bypasses it drops Ollama back onto 0.0.0.0:11434 — the address
# the negotiator already owns. This helper re-asserts the pin and restarts the
# pair in the only safe order: proxy down, backend up, proxy up.
set -euo pipefail

STATE="/usr/local/share/ollama-unify/state.env"
DROPIN="/etc/systemd/system/ollama.service.d/zzz-ollama-unify-safety.conf"
NEG_CONF="/etc/default/ollama-unify-negotiator"

log() { printf 'ollama-unify-reconcile: %s\n' "$*"; }

[ -r "$STATE" ] || { log "no installed state at $STATE; nothing to reconcile"; exit 0; }
# shellcheck disable=SC1090
. "$STATE"

BACKEND="${OLLAMA_UNIFY_PINNED_BACKEND:-}"
[ -n "$BACKEND" ] || { log "state carries no pinned backend; nothing to reconcile"; exit 0; }

changed=0

current_version="$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
if [ -n "$current_version" ] && [ "$current_version" != "${OLLAMA_UNIFY_OLLAMA_VERSION:-}" ]; then
  log "Ollama moved ${OLLAMA_UNIFY_OLLAMA_VERSION:-unknown} -> $current_version"
  changed=1
fi

if [ ! -r "$DROPIN" ]; then
  log "safety drop-in missing; restoring the backend pin only"
  mkdir -p "$(dirname "$DROPIN")"
  printf '# Restored by ollama-unify-reconcile\n[Service]\nEnvironment="OLLAMA_HOST=%s"\n' \
    "$BACKEND" > "$DROPIN"
  log "run 'ollama-unify.sh --install-safety' to restore the full containment policy"
  changed=1
elif ! grep -qF "OLLAMA_HOST=${BACKEND}" "$DROPIN"; then
  log "drop-in OLLAMA_HOST drifted; repinning to $BACKEND"
  if grep -qE '^Environment="OLLAMA_HOST=' "$DROPIN"; then
    sed -i -E "s|^Environment=\"OLLAMA_HOST=.*\"|Environment=\"OLLAMA_HOST=${BACKEND}\"|" "$DROPIN"
  else
    printf 'Environment="OLLAMA_HOST=%s"\n' "$BACKEND" >> "$DROPIN"
  fi
  changed=1
fi

if [ -w "$NEG_CONF" ] && ! grep -qF "OLLAMA_UNIFY_BACKEND=\"${BACKEND}\"" "$NEG_CONF"; then
  log "negotiator backend drifted; repinning to $BACKEND"
  sed -i -E "s|^OLLAMA_UNIFY_BACKEND=.*|OLLAMA_UNIFY_BACKEND=\"${BACKEND}\"|" "$NEG_CONF"
  changed=1
fi

if [ "$changed" = 0 ]; then
  log "no drift detected"
  exit 0
fi

systemctl daemon-reload
systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
systemctl restart ollama.service
if systemctl list-unit-files ollama-unify-negotiator.service >/dev/null 2>&1; then
  systemctl start ollama-unify-negotiator.service || log "negotiator failed to start"
fi

if [ -n "$current_version" ]; then
  sed -i -E "s|^OLLAMA_UNIFY_OLLAMA_VERSION=.*|OLLAMA_UNIFY_OLLAMA_VERSION=\"${current_version}\"|" "$STATE" || true
fi
log "reconciled; Ollama pinned to $BACKEND"
RECONCILE
}

render_reconcile_units() {
  local binary="$1"
  cat <<UNIT
# Managed by ollama-unify — generated $(date '+%Y-%m-%dT%H:%M:%S%z')
[Unit]
Description=Repin Ollama behind the ollama-unify negotiator after an upgrade
Documentation=https://github.com/robit-man/ollama-unify
After=ollama.service

[Service]
Type=oneshot
ExecStart=$SAFETY_RECONCILE_HELPER_PATH
UNIT
}

render_reconcile_path_unit() {
  local binary="$1"
  cat <<UNIT
# Managed by ollama-unify — generated $(date '+%Y-%m-%dT%H:%M:%S%z')
[Unit]
Description=Watch the Ollama binary for upgrades that clear the pinned backend
Documentation=https://github.com/robit-man/ollama-unify

[Path]
PathChanged=$binary
Unit=ollama-unify-reconcile.service

[Install]
WantedBy=multi-user.target
UNIT
}

install_reconcile_watchdog() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  local binary helper_dir
  binary=$(ollama_binary_path || printf '')
  helper_dir="${SAFETY_RECONCILE_HELPER_PATH%/*}"

  "${elevate[@]}" mkdir -p "$helper_dir" "$SAFETY_DISCOVERY_DIR"
  render_reconcile_helper | "${elevate[@]}" tee "$SAFETY_RECONCILE_HELPER_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_RECONCILE_HELPER_PATH"
  render_reconcile_state | "${elevate[@]}" tee "$SAFETY_STATE_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_STATE_PATH"
  ok "reconcile helper installed: $SAFETY_RECONCILE_HELPER_PATH"

  if [ -z "$binary" ]; then
    warn "the Ollama binary could not be located; the upgrade watchdog was not installed"
    return 0
  fi
  render_reconcile_units "$binary" | "${elevate[@]}" tee "$SAFETY_RECONCILE_SERVICE_PATH" >/dev/null
  render_reconcile_path_unit "$binary" | "${elevate[@]}" tee "$SAFETY_RECONCILE_PATH_UNIT_PATH" >/dev/null
  "${elevate[@]}" systemctl daemon-reload
  "${elevate[@]}" systemctl enable --now ollama-unify-reconcile.path >/dev/null 2>&1 \
    || warn "ollama-unify-reconcile.path could not be enabled"
  ok "upgrade watchdog armed on $binary"
}

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
    --check-update)
      detect_host_profile
      check_ollama_update
      exit 0
      ;;
    --reconcile)
      detect_host_profile
      reconcile_after_update
      exit 0
      ;;
    --update-ollama)
      detect_host_profile
      update_ollama
      exit 0
      ;;
    -h|--help)
      say "Usage: ./ollama-unify.sh [--classify|--safety-preview|--print-env|--install-safety]"
      say "                         [--check-update|--reconcile|--update-ollama]"
      say "  --classify        Classify the host, accelerators, backend, and risk tier without changes."
      say "  --safety-preview  Classify the host and print the generated safety policy without changes."
      say "  --print-env       Print shell exports for the selected scheduler/backend policy."
      say "  --install-safety  Install systemd containment and the dynamic GPU negotiator; do not migrate models."
      say "  --check-update    Report the installed vs latest Ollama release and any pinning drift."
      say "  --reconcile       Repin Ollama behind the negotiator after an out-of-band Ollama upgrade."
      say "  --update-ollama   Update Ollama inside a safe stop → repin → start envelope, then verify."
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
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1; then
    $SUDO systemctl stop ollama-unify-negotiator.service
    ok "ollama-unify-negotiator.service stopped"
  fi
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
      if systemctl is-enabled ollama-unify-negotiator.service >/dev/null 2>&1; then
        $SUDO systemctl start ollama-unify-negotiator.service || true
      fi
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
