#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture_bin="$repo_dir/tests/fixtures/bin"
script="$repo_dir/ollama-unify.sh"
test_path="$fixture_bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Keep classification fixtures independent of models and journals on the host
# running the test. Individual cases override these measurements when needed.
export OLLAMA_SAFE_LARGEST_MODEL_MIB=4096
export OLLAMA_SAFE_OBSERVED_HOST_MIB=2048

assert_contains() {
  local output="$1" expected="$2"
  if [[ "$output" != *"$expected"* ]]; then
    printf 'FAIL: expected output to contain: %s\n' "$expected" >&2
    exit 1
  fi
}

assert_not_contains() {
  local output="$1" unexpected="$2"
  if [[ "$output" == *"$unexpected"* ]]; then
    printf 'FAIL: output unexpectedly contained: %s\n' "$unexpected" >&2
    exit 1
  fi
}

cuda_output=$(PATH="$test_path" MOCK_PROFILE=cuda OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=65536 "$script" --safety-preview)
assert_contains "$cuda_output" '[cuda/dedicated] GPU 0: Mock CUDA 24GB'
assert_contains "$cuda_output" '[cuda/shared-display] GPU 1: Mock CUDA Display 12GB'
assert_contains "$cuda_output" '[cuda/constrained] GPU 2: Mock CUDA 2GB'
assert_contains "$cuda_output" 'Backend: cuda (dedicated)'
assert_contains "$cuda_output" 'CUDA_VISIBLE_DEVICES=GPU-dedicated'
assert_contains "$cuda_output" 'Scheduler: 1 model(s), 1 parallel request(s)'
assert_not_contains "$(printf '%s\n' "$cuda_output" | grep CUDA_VISIBLE_DEVICES)" 'GPU-display'

triple_cuda_output=$(PATH="$test_path" MOCK_PROFILE=cuda_triple OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=289468 \
  OLLAMA_SAFE_LARGEST_MODEL_MIB=106217 OLLAMA_SAFE_OBSERVED_HOST_MIB=50434 \
  "$script" --safety-preview)
assert_contains "$triple_cuda_output" 'Aggregate dedicated device memory: 245760 MiB across 3 accelerator(s); 84% of host RAM'
assert_contains "$triple_cuda_output" 'Model scan: largest installed inference payload 106217 MiB (explicit override)'
assert_contains "$triple_cuda_output" 'Ollama history: largest observed host projection 50434 MiB'
assert_contains "$triple_cuda_output" 'Host memory: throttle at 50434 MiB; hard cap at 106217 MiB; 183251 MiB remains outside the cgroup'
assert_contains "$triple_cuda_output" 'MemoryHigh=50434M'
assert_contains "$triple_cuda_output" 'MemoryMax=106217M'
assert_contains "$triple_cuda_output" 'MemorySwapMax=0'
assert_contains "$triple_cuda_output" 'GPU policy: native live-VRAM placement; forced spread disabled'
assert_contains "$triple_cuda_output" 'GPU negotiator: cooperative leases plus anonymous-process rebalance'
assert_contains "$triple_cuda_output" 'OLLAMA_SCHED_SPREAD=0'
assert_contains "$triple_cuda_output" 'CUDA_VISIBLE_DEVICES=GPU-large-0,GPU-large-1,GPU-large-2'
assert_contains "$triple_cuda_output" 'OLLAMA_HOST=127.0.0.1:11436'
assert_contains "$triple_cuda_output" 'LLAMA_ARG_N_GPU_LAYERS=auto'
assert_contains "$triple_cuda_output" 'LLAMA_ARG_SPLIT_MODE=layer'
assert_contains "$triple_cuda_output" 'LLAMA_ARG_FIT=on'
assert_contains "$triple_cuda_output" 'GGML_CUDA_NO_PINNED=1'
assert_contains "$triple_cuda_output" 'UnsetEnvironment=GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_REGISTER_HOST LLAMA_ARG_FIT_TARGET'
assert_not_contains "$triple_cuda_output" 'Environment="GGML_CUDA_ENABLE_UNIFIED_MEMORY='
assert_not_contains "$triple_cuda_output" 'OLLAMA_GPU_OVERHEAD='

mixed_cuda_output=$(PATH="$test_path" MOCK_PROFILE=cuda_mixed OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=196608 \
  OLLAMA_SAFE_LARGEST_MODEL_MIB=106217 OLLAMA_SAFE_OBSERVED_HOST_MIB=50434 \
  "$script" --safety-preview)
assert_contains "$mixed_cuda_output" 'Aggregate dedicated device memory: 131072 MiB across 2 accelerator(s); 66% of host RAM'
assert_contains "$mixed_cuda_output" 'Host memory: throttle at 50434 MiB; hard cap at 106217 MiB; 90391 MiB remains outside the cgroup'

display_output=$(PATH="$test_path" MOCK_PROFILE=cuda_display OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=32768 "$script" --safety-preview)
assert_contains "$display_output" 'Backend: cuda (shared-display)'
assert_contains "$display_output" 'CUDA_VISIBLE_DEVICES=GPU-display'
assert_contains "$display_output" 'Device memory: live free-VRAM telemetry; no guessed fixed carve-out'
assert_not_contains "$display_output" 'LLAMA_ARG_N_GPU_LAYERS=auto'

rocm_output=$(PATH="$test_path" MOCK_PROFILE=rocm OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=131072 "$script" --safety-preview)
assert_contains "$rocm_output" '[rocm/discrete] GPU 0: Mock AMD 48GB'
assert_contains "$rocm_output" 'Backend: rocm (discrete)'
assert_contains "$rocm_output" 'ROCR_VISIBLE_DEVICES=GPU-mock-amd-0'
rocm_env=$(PATH="$test_path" MOCK_PROFILE=rocm OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=131072 "$script" --print-env)
assert_contains "$rocm_env" 'unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL'
assert_contains "$rocm_env" 'export ROCR_VISIBLE_DEVICES=GPU-mock-amd-0\,GPU-mock-amd-1'
assert_contains "$rocm_env" 'export LLAMA_ARG_N_GPU_LAYERS=auto'

vulkan_output=$(PATH="$test_path" MOCK_PROFILE=vulkan OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=16384 "$script" --safety-preview)
assert_contains "$vulkan_output" '[vulkan/shared] GPU 0: Mock Integrated Vulkan GPU'
assert_contains "$vulkan_output" 'Backend: vulkan (shared/integrated)'
assert_contains "$vulkan_output" 'GGML_VK_VISIBLE_DEVICES=0'
assert_contains "$vulkan_output" 'OLLAMA_VULKAN=1'

metal_output=$(PATH="$test_path" MOCK_PROFILE=metal OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=32768 "$script" --safety-preview 2>&1)
assert_contains "$metal_output" 'Platform: macOS 15.0-mock (Darwin/arm64; none)'
assert_contains "$metal_output" '[metal/unified] Apple Mock GPU'
assert_contains "$metal_output" 'Backend: metal (unified-memory)'
assert_contains "$metal_output" 'Native cgroup OOM containment is unavailable under launchd'

cpu_output=$(PATH="$test_path" MOCK_PROFILE=cpu OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=6144 "$script" --safety-preview)
assert_contains "$cpu_output" 'Class: constrained'
assert_contains "$cpu_output" 'Backend: cpu (host-memory)'
assert_contains "$cpu_output" '2048-token context, queue 8'
assert_contains "$cpu_output" 'CUDA_VISIBLE_DEVICES=-1'
assert_not_contains "$cpu_output" 'OLLAMA_GPU_OVERHEAD='
assert_not_contains "$cpu_output" 'GGML_CUDA_NO_PINNED=1'
assert_not_contains "$cpu_output" 'GPU negotiator:'
assert_not_contains "$cpu_output" 'OLLAMA_HOST=127.0.0.1:11436'

if PATH="$test_path" MOCK_PROFILE=cpu OLLAMA_SAFE_BACKEND=bogus "$script" --classify >/dev/null 2>&1; then
  printf 'FAIL: invalid backend override succeeded\n' >&2
  exit 1
fi

legacy_systemd=$(PATH="$test_path" MOCK_PROFILE=cpu OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=16384 \
  bash -c 'source "$1"; build_safety_profile; SYSTEMD_VERSION=230; render_safety_service_directives' _ "$script")
assert_not_contains "$legacy_systemd" 'MemoryMax='
assert_not_contains "$legacy_systemd" 'OOMPolicy='
assert_not_contains "$legacy_systemd" 'ManagedOOMMemoryPressure='
assert_contains "$legacy_systemd" 'ExecStartPre=/usr/local/libexec/ollama-unify-memory-preflight'

modern_systemd=$(PATH="$test_path" MOCK_PROFILE=cpu OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=16384 \
  bash -c 'source "$1"; build_safety_profile; SYSTEMD_VERSION=255; render_safety_service_directives' _ "$script")
assert_contains "$modern_systemd" 'MemoryMax='
assert_contains "$modern_systemd" 'MemorySwapMax='
assert_contains "$modern_systemd" 'OOMPolicy=stop'
assert_contains "$modern_systemd" 'CPUQuota=400%'
assert_contains "$modern_systemd" 'CPUWeight=10'
assert_contains "$modern_systemd" 'IOWeight=10'
assert_contains "$modern_systemd" 'ManagedOOMMemoryPressure=kill'
assert_contains "$modern_systemd" 'ManagedOOMMemoryPressureLimit=20%'
assert_contains "$modern_systemd" 'ManagedOOMSwap=kill'
assert_contains "$modern_systemd" 'ExecCondition=/usr/local/libexec/ollama-unify-memory-preflight'
assert_not_contains "$modern_systemd" 'ExecStartPre=/usr/local/libexec/ollama-unify-memory-preflight'
assert_contains "$modern_systemd" 'Restart=no'

override_systemd=$(PATH="$test_path" MOCK_PROFILE=cpu OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=16384 \
  OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT=35 OLLAMA_SAFE_CPU_QUOTA_PERCENT=200 \
  OLLAMA_SAFE_CPU_WEIGHT=25 OLLAMA_SAFE_IO_WEIGHT=30 OLLAMA_SAFE_RESTART_POLICY=on-failure \
  bash -c 'source "$1"; build_safety_profile; SYSTEMD_VERSION=255; render_safety_service_directives' _ "$script")
assert_contains "$override_systemd" 'ManagedOOMMemoryPressureLimit=35%'
assert_contains "$override_systemd" 'CPUQuota=200%'
assert_contains "$override_systemd" 'CPUWeight=25'
assert_contains "$override_systemd" 'IOWeight=30'
assert_contains "$override_systemd" 'Restart=on-failure'

if PATH="$test_path" MOCK_PROFILE=cpu OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT=101 \
  "$script" --classify >/dev/null 2>&1; then
  printf 'FAIL: invalid pressure threshold succeeded\n' >&2
  exit 1
fi

model_fixture=$(mktemp -d)
mkdir -p "$model_fixture/manifests/registry.ollama.ai/library/measured"
printf '%s\n' '{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:model","size":2147483648},{"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:projector","size":1073741824},{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:template","size":999999999}]}' \
  > "$model_fixture/manifests/registry.ollama.ai/library/measured/latest"
measured_output=$(env -u OLLAMA_SAFE_LARGEST_MODEL_MIB PATH="$test_path" MOCK_PROFILE=cpu \
  OLLAMA_SAFE_MODEL_STORE="$model_fixture" OLLAMA_SAFE_OBSERVED_HOST_MIB=512 \
  OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=8192 "$script" --classify)
assert_contains "$measured_output" 'Model scan: largest installed inference payload 3072 MiB'
assert_contains "$measured_output" 'Host memory: throttle at 512 MiB; hard cap at 3072 MiB; 5120 MiB remains outside the cgroup'

empty_fixture=$(mktemp -d)
mkdir -p "$empty_fixture/manifests"
if env -u OLLAMA_SAFE_LARGEST_MODEL_MIB PATH="$test_path" MOCK_PROFILE=cpu \
  OLLAMA_SAFE_MODEL_STORE="$empty_fixture" OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=8192 \
  "$script" --classify >/dev/null 2>&1; then
  printf 'FAIL: classifier invented a host limit without a model payload\n' >&2
  exit 1
fi

preflight_script=$(mktemp)
trap 'rm -f "$preflight_script"; rm -rf "$model_fixture" "$empty_fixture"' EXIT
bash -c 'source "$1"; render_safety_preflight_script' _ "$script" > "$preflight_script"
chmod +x "$preflight_script"
if "$preflight_script" 999999999 20 >/dev/null 2>&1; then
  printf 'FAIL: memory preflight accepted an impossible reserve\n' >&2
  exit 1
else
  preflight_status=$?
  if [ "$preflight_status" -ne 75 ]; then
    printf 'FAIL: memory preflight returned %s instead of 75\n' "$preflight_status" >&2
    exit 1
  fi
fi

printf 'classifier fixtures: PASS (CUDA dedicated/display, ROCm, Vulkan, Metal, CPU)\n'
