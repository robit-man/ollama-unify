# WO-03 client-history TTL and tray screen bounds

## Objective

Stop historical broker client/lane records from compounding for the lifetime
of the daemon, and guarantee that every rendered tray menu label fits within
the narrowest active monitor work area.

## Root-cause anchors

- `ollama-unify.sh` — generated negotiator constants and `Broker.clients`.
- `Broker.record_client_use` — currently caps only the number of client keys.
- `Broker._client_summaries_locked` — currently exports stale clients and an
  unbounded per-client lane map.
- `TrayApp.apply_row` — currently sends unconstrained labels to GTK.
- `install_gpu_negotiator` — owns durable environment defaults.

## Implementation checklist

- [x] Add a configurable client-history TTL with a safe default.
- [x] Prune expired clients both when recording traffic and when status is read.
- [x] Preserve clients involved in an in-flight lane while pruning.
- [x] Bound and TTL-prune each client's historical lane records.
- [x] Apply the same TTL to every lane's historical `Used by` records.
- [x] Renew attribution expiry automatically on each admitted request.
- [x] Publish effective retention policy and row expiry in status/discovery.
- [x] Install the retention defaults in the broker environment file.
- [x] Derive the tray label budget from the narrowest monitor work area.
- [x] Pixel-measure and ellipsize every non-separator tray label.
- [x] Preserve the full label in a tooltip when its visible form is shortened.
- [x] Document the retention policy and tray width behavior.

## Verification checklist

- [x] Regression proves an inactive client disappears after TTL.
- [x] Regression proves per-client ended-lane history stays bounded.
- [x] Regression proves an in-flight client survives TTL until completion.
- [x] Regression proves per-lane `Used by` history expires after completion.
- [x] Regression proves label shortening fits its pixel budget.
- [x] Generated negotiator and tray scripts compile.
- [x] Full broker and tray test suite passes.
- [x] Classifier and transfer fixture suites pass.
- [x] Shell/Python static checks and `git diff --check` pass.
- [x] Source worktree is clean after commit.
- [x] Commit is pushed to the tracked remote branch.

## Deployment checklist

- [x] Re-run `docker gpu discover` immediately before service mutation.
- [x] Snapshot the active lease owners/scopes/states without exposing tokens.
- [x] Install generated broker/tray artifacts from this source checkout.
- [x] Restart the negotiator and user tray service.
- [x] Verify the negotiated API, control socket, and services are healthy.
- [x] Verify active leases retain their owner, scope, and state.
- [x] Verify installed retention metadata reflects the new defaults.
- [x] Verify installed tray labels are screen-bounded.

## Rollback

Restore the previous generated negotiator and tray artifacts (or check out the
previous commit and run `./ollama-unify.sh --install-safety`), reload systemd,
then restart `ollama.service`, `ollama-unify-negotiator.service`, and the user
tray service in that order. Re-run discovery and compare the lease snapshot.

## Verification evidence

- 2026-10-02: `bash tests/test-negotiator.sh` passed, including real GTK label
  measurement and the client TTL/renewal/in-flight/lane-bound integration case.
- 2026-10-02: `bash tests/test-classifier.sh` passed.
- 2026-10-02: `bash tests/test-transfer.sh` passed.
- 2026-10-02: `bash -n`, `shellcheck`, `ruff check`, and `git diff --check`
  passed.
- 2026-10-02: repair commit `43b4b98` pushed to `origin/main`.
- 2026-10-02: pre-deployment discovery reported the broker healthy and one
  active Voryn lease scoped to GPU
  `GPU-170a99ee-850f-2182-1050-4e8d3c87b6b0`. A token-free status snapshot
  and recoverable installed-artifact copy were written under `/tmp`.
- 2026-10-02: `./ollama-unify.sh --install-safety` installed the generated
  negotiator and tray. Its immediate discovery refresh raced startup once;
  subsequent API, socket, service, and discovery checks all passed.
- 2026-10-02: installed negotiator and tray SHA-256 values exactly matched the
  source render. The negotiator, Ollama backend, and reloaded/restarted user
  tray were active with no warning-or-higher journal entries after deployment.
- 2026-10-02: the Voryn lease retained its owner, active state, exact GPU scope,
  and renewed heartbeat. Live discovery reports a 3600-second client-history
  TTL, 256-client cap, 32 lanes per client, and 8 clients per lane.
