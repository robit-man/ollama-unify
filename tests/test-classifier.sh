#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture_bin="$repo_dir/tests/fixtures/bin"
script="$repo_dir/ollama-unify.sh"
test_path="$fixture_bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

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
  "$script" --safety-preview)
assert_contains "$triple_cuda_output" 'Aggregate dedicated device memory: 245760 MiB across 3 accelerators'
assert_contains "$triple_cuda_output" 'Host memory: reserve 28946 MiB; throttle at 245760 MiB; hard cap at 260522 MiB'
assert_contains "$triple_cuda_output" 'MemoryHigh=245760M'
assert_contains "$triple_cuda_output" 'MemoryMax=260522M'

mixed_cuda_output=$(PATH="$test_path" MOCK_PROFILE=cuda_mixed OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=196608 \
  "$script" --safety-preview)
assert_contains "$mixed_cuda_output" 'Aggregate dedicated device memory: 131072 MiB across 2 accelerators'
assert_contains "$mixed_cuda_output" 'Host memory: reserve 19660 MiB; throttle at 131072 MiB; hard cap at 176948 MiB'

display_output=$(PATH="$test_path" MOCK_PROFILE=cuda_display OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=32768 "$script" --safety-preview)
assert_contains "$display_output" 'Backend: cuda (shared-display)'
assert_contains "$display_output" 'CUDA_VISIBLE_DEVICES=GPU-display'
assert_contains "$display_output" 'Device-memory reserve: 2457 MiB'

rocm_output=$(PATH="$test_path" MOCK_PROFILE=rocm OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=131072 "$script" --safety-preview)
assert_contains "$rocm_output" '[rocm/discrete] GPU 0: Mock AMD 48GB'
assert_contains "$rocm_output" 'Backend: rocm (discrete)'
assert_contains "$rocm_output" 'ROCR_VISIBLE_DEVICES=GPU-mock-amd-0'
rocm_env=$(PATH="$test_path" MOCK_PROFILE=rocm OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB=131072 "$script" --print-env)
assert_contains "$rocm_env" 'unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL'
assert_contains "$rocm_env" 'export ROCR_VISIBLE_DEVICES=GPU-mock-amd-0\,GPU-mock-amd-1'

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

preflight_script=$(mktemp)
trap 'rm -f "$preflight_script"' EXIT
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
