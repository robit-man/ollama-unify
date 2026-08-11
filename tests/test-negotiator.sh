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

python3 -m py_compile "$test_tmp/ollama-unify-gpu-negotiator"
"$test_tmp/ollama-unify-gpu-negotiator" self-test
python3 "$repo_dir/tests/test-negotiator.py" \
  "$test_tmp/ollama-unify-gpu-negotiator" "$repo_dir/tests/fixtures/bin"
