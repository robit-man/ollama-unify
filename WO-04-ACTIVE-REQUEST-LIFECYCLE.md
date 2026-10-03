# WO-04 active-request lifecycle leases

## Objective

Prevent a disconnected client or silent Ollama backend from pinning a broker
admission forever, and stop leaked `in_flight` counters from causing duplicate
managed lanes and false VRAM exhaustion.

## Live failure anchors

- Before this repair, `ProxyHandler._handle` created the backend
  `HTTPConnection` with `timeout=None`, then blocked in `getresponse()` or
  response reads.
- Client connectivity is checked during admission and once immediately after
  admission, but not while the backend request is executing.
- `Broker.proxy_exit` is the only normal decrement path for both the lane and
  global active-request counts.
- `Broker.capacity_reconciler` treats every leaked same-model admission as
  real parallel demand and can start another fully reserved lane.
- `Broker.pool_reaper` intentionally excludes every lane whose `in_flight`
  count is nonzero, so it cannot repair leaked admissions.
- Live incident evidence showed three ready 27B lanes, each permanently at
  `in_flight=1`, while the physical target GPU still had ample free VRAM.

## Implementation checklist

- [x] Represent every admission with an exact, unique active-request record.
- [x] Make release idempotent by admission identity rather than raw counters.
- [x] Track admitted, backend-active, detached, cancelling, and terminal state.
- [x] Give terminal finalization a bounded lease and fail-closed tombstone.
- [x] Give process-group stop attempts a bounded, renewable lease.
- [x] Add a configurable backend inactivity TTL with a safe installed default.
- [x] Renew the activity TTL on backend connection, headers, and every chunk.
- [x] Detect client EOF/reset during inference through an event-driven watcher.
- [x] Cancel non-replayable work immediately when its client disconnects.
- [x] Give replayable logical work a short, renewable detached completion TTL.
- [x] Close the backend transport when a request lease expires.
- [x] Retire a managed lane if transport cancellation does not terminate work.
- [x] Block fresh same-model admission while cancellation is unresolved.
- [x] Release an expired admission only after its backend lane is stopped.
- [x] Keep retiring/stubborn process-group reservations charged until exit.
- [x] Escalate termination against the complete runner process group.
- [x] Prevent late handler cleanup from decrementing a newer admission.
- [x] Publish lifecycle TTLs, phases, ages, and aggregate expiry metrics.
- [x] Install and document the lifecycle defaults and supported overrides.

## Verification checklist

- [x] Regression proves repeated abandoned requests do not compound lanes.
- [x] Regression proves backend chunks renew a request beyond its base TTL.
- [x] Regression preserves quick logical completion and byte replay after RST.
- [x] Regression proves a silent backend expires and releases exact ownership.
- [x] Regression proves late cleanup cannot decrement a newer active request.
- [x] Regression proves cancellation retires a shared parallel lane before a
  follower can reach either the suspect lane or a replacement lane.
- [x] Regression proves cancellation-induced EOF cannot enter replay storage.
- [x] Regression proves client EOF after a complete fixed-length body cannot
  race completion and retire a healthy lane.
- [x] Regression proves truncated fixed-length output closes the client and
  retires its lane.
- [x] Regression proves backend reset retires/replaces the suspect lane.
- [x] Regression proves stubborn process-group children receive SIGKILL and a
  failed stop retains its reservation in placement accounting.
- [x] Regression proves stalled terminal finalization has bounded cleanup.
- [x] Generated negotiator script compiles.
- [x] Full broker, classifier, transfer, tray, and shell/static suites pass.
- [x] `git diff --check` passes.
- [ ] Repair is committed and pushed to the tracked remote branch.

## Deployment checklist

- [ ] Run `docker gpu discover` immediately before service mutation.
- [ ] Snapshot active lease owners, scopes, and states without lease tokens.
- [ ] Install generated artifacts from this exact source checkout.
- [ ] Restart the broker without restarting or reallocating external workloads.
- [ ] Verify API health, control socket health, and active-lease preservation.
- [ ] Verify stale active admissions and duplicate managed lanes are gone.
- [ ] Verify installed lifecycle policy matches source defaults.
- [ ] Observe fresh Omnius traffic without request or lane accumulation.

## Rollback

Restore the previous generated negotiator artifact (or check out the prior
commit and run `./ollama-unify.sh --install-safety`), reload systemd, and
restart `ollama-unify-negotiator.service`. Re-run discovery and compare the
token-free lease snapshot before allowing new inference traffic.

## Verification evidence

- 2026-10-03: targeted lifecycle and terminal-race cases passed, including
  three consecutive runs of the renewable/expiry/abandonment group and ten
  consecutive runs of the prior client-history/lane-stop race.
- 2026-10-03: focused shared-parallel cancellation, cancellation-induced EOF,
  truncated response, backend-reset replacement, stubborn process-group, and
  bounded terminal-finalization regressions passed.
- 2026-10-03: the fixed-length completion/EOF race passed five consecutive
  isolated queue-bypass runs, followed by a clean full integration run.
- 2026-10-03: `bash tests/test-negotiator.sh` passed the full broker, pool,
  replay, queue, lease, client identity, and tray integration matrix.
- 2026-10-03: `bash tests/test-classifier.sh` and
  `bash tests/test-transfer.sh` passed.
- 2026-10-03: generated negotiator/tray compilation, `ruff check`, `bash -n`,
  `shellcheck`, and `git diff --check` passed.
- Commit, push, and live deployment verification remain pending.
