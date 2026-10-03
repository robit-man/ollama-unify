# WO-05 GPU constraint reconciliation

## Objective

Keep an explicit per-request GPU allowlist hard while allowing the host's live
selected GPU set to shrink. A stale unavailable member must not reject a
request when another allowed member remains available, and it must never cause
retry amplification in clients.

## Incident anchors

- Omnius sent `X-Ollama-Unify-GPU-UUIDs` with two selected A100 UUIDs plus a
  formerly selected UUID that is no longer healthy.
- The broker rejected the entire allowlist with non-retryable HTTP 422 even
  though its intersection with the selected set contained two valid targets.
- Omnius then erased the permanent failure classification and amplified the
  rejection into eight attempts: JSON plus plain retry across the original
  request and three durable Telegram recoveries.

## Broker implementation checklist

- [x] Treat the request list as an ordered set-membership constraint, not a
  requirement that every historical member remain installed.
- [x] Preserve caller order while intersecting with live broker-selected GPUs.
- [x] Never route outside the caller's explicit allowlist.
- [x] Reject empty input and a nonempty allowlist with an empty live
  intersection.
- [x] Log pruned unavailable UUIDs and the effective allowlist.
- [x] Publish the reconciliation semantics in discovery and operator docs.
- [x] Cover capacity and ordinary inference with mixed live/stale UUIDs.
- [x] Preserve the all-unavailable non-retryable rejection regression.

## Omnius implementation checklist

- [x] Preserve typed broker failure metadata across Telegram router layers.
- [x] Skip JSON-to-plain fallback for `retryable:false` broker failures.
- [x] Do not schedule durable Telegram recovery for a permanent failure.
- [x] Prove one wire attempt and zero scheduled recoveries for broker 422
  `gpu_constraint_unavailable`.

## Delivery and live validation checklist

- [x] Full broker integration/static suite passes.
- [x] Broker repair is committed and pushed to the tracked branch.
- [x] Run `docker gpu discover` immediately before service mutation.
- [x] Install generated broker/tray artifacts from the committed checkout.
- [x] Restart broker and tray while preserving active external leases.
- [x] Verify installed discovery advertises intersection semantics.
- [x] Verify a mixed selected/stale request constraint resolves to selected
  members on a metadata request without issuing model inference.
- [ ] Observe the user's next Omnius request route past broker admission with
  no repeated 422s.
- [x] Omnius permanent-failure repair passes focused tests, is committed, and
  is pushed to its tracked branch.

## Rollback

Restore the prior broker commit and rerun `./ollama-unify.sh --install-safety`.
The old behavior fails closed on any unavailable list member, so clients must
then manually remove stale UUIDs before inference can resume.
