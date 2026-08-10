#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/ollama-unify.sh"

fixture=$(mktemp -d)
trap 'rm -rf -- "$fixture"' EXIT

mkdir -p "$fixture/mv-src/manifests/registry/model" "$fixture/mv-src/blobs" \
  "$fixture/mv-dst/manifests" "$fixture/mv-dst/blobs"
touch "$fixture/mv-src/manifests/registry/model/latest" "$fixture/mv-src/blobs/sha256-mock"
transfer_store "$fixture/mv-src" "$fixture/mv-dst" mv "" >/dev/null
test -f "$fixture/mv-dst/manifests/registry/model/latest"
test -f "$fixture/mv-dst/blobs/sha256-mock"

mkdir -p "$fixture/rsync-src/manifests/registry/model" "$fixture/rsync-src/blobs" "$fixture/rsync-dst"
touch "$fixture/rsync-src/manifests/registry/model/latest" "$fixture/rsync-src/blobs/sha256-mock"
transfer_store "$fixture/rsync-src" "$fixture/rsync-dst" rsync "" >/dev/null
test -f "$fixture/rsync-dst/manifests/registry/model/latest"
test -f "$fixture/rsync-dst/blobs/sha256-mock"

expected=$(cd "$fixture/rsync-dst" && pwd -P)
actual=$(canonical_path "$fixture/rsync-dst")
test "$actual" = "$expected"

printf 'transfer fixtures: PASS (same-filesystem move, rsync, canonical paths)\n'
