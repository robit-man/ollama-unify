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
- [ ] Source worktree is clean after commit.
- [ ] Commit is pushed to the tracked remote branch.

## Deployment checklist

- [ ] Re-run `docker gpu discover` immediately before service mutation.
- [ ] Snapshot the active lease owners/scopes/states without exposing tokens.
- [ ] Install generated broker/tray artifacts from this source checkout.
- [ ] Restart the negotiator and user tray service.
- [ ] Verify the negotiated API, control socket, and services are healthy.
- [ ] Verify active leases retain their owner, scope, and state.
- [ ] Verify installed retention metadata reflects the new defaults.
- [ ] Verify installed tray labels are screen-bounded.

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
