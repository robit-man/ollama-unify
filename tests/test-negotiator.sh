#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test_tmp=$(mktemp -d)
trap 'rm -rf "$test_tmp"' EXIT

listen=$(
  # shellcheck disable=SC1091
  source "$repo_dir/ollama-unify.sh"
  OLLAMA_SAFE_NEGOTIATOR_LISTEN=0.0.0.0:11434 detect_ollama_proxy_listen
)
[ "$listen" = "0.0.0.0:11434" ] || { printf 'invalid normalized listen address: %s\n' "$listen" >&2; exit 1; }

(
  # Resolved from the test's repository root.
  # shellcheck disable=SC1091
  source "$repo_dir/ollama-unify.sh"
  render_gpu_negotiator_script
) > "$test_tmp/ollama-unify-gpu-negotiator"
chmod 0755 "$test_tmp/ollama-unify-gpu-negotiator"

(
  # shellcheck disable=SC1091
  source "$repo_dir/ollama-unify.sh"
  render_docker_gpu_lease_plugin
) > "$test_tmp/docker-gpu"
chmod 0755 "$test_tmp/docker-gpu"
plugin_metadata=$("$test_tmp/docker-gpu" docker-cli-plugin-metadata)
[[ "$plugin_metadata" == *'"Vendor":"ollama-unify"'* ]] \
  || { printf 'invalid Docker plugin metadata: %s\n' "$plugin_metadata" >&2; exit 1; }
plugin_discovery=$(OLLAMA_UNIFY_GPU_LEASE_CLI="$test_tmp/ollama-unify-gpu-negotiator" \
  MOCK_PROFILE=cuda_triple PATH="$repo_dir/tests/fixtures/bin:$PATH" \
  "$test_tmp/docker-gpu" gpu discover)
[[ "$plugin_discovery" == *'"selected_gpu_count"'* ]] \
  || { printf 'Docker plugin did not forward its prefixed invocation\n' >&2; exit 1; }

agent_instructions=$(
  # shellcheck disable=SC1091
  source "$repo_dir/ollama-unify.sh"
  render_global_codex_gpu_block
)
[[ "$agent_instructions" == *'docker gpu discover'* ]] \
  || { printf 'global agent instructions lack Docker discovery command\n' >&2; exit 1; }

python3 -m py_compile "$test_tmp/ollama-unify-gpu-negotiator"
python3 -m py_compile "$repo_dir/tests/test-negotiator.py" \
  "$repo_dir/tests/test-negotiator-pool.py" "$repo_dir/tests/fixtures/bin/ollama"
"$test_tmp/ollama-unify-gpu-negotiator" self-test
python3 "$repo_dir/tests/test-negotiator.py" \
  "$test_tmp/ollama-unify-gpu-negotiator" "$repo_dir/tests/fixtures/bin"
python3 "$repo_dir/tests/test-negotiator-pool.py" \
  "$test_tmp/ollama-unify-gpu-negotiator" "$repo_dir/tests/fixtures/bin"
