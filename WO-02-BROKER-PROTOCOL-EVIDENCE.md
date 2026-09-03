# WO-02 broker protocol evidence

## Closed protocol gaps

- A bounded admission timeout returns a current, content-free queue position
  and a stable opaque queue ticket.
- The broker retains the accepted and clamped request body under the logical
  request ID. A reconnect sends only the ID and the typed resume header.
- One resumed request keeps its original queue entry, broker request ID, and
  ticket. It does not enqueue or generate a second request.
- Completed native and streaming responses replay byte for byte without a
  second backend request.
- Cancelled and expired retained requests leave typed, non-retryable
  tombstones for body-free resume. A full-body request remains an explicit new
  attempt after expiry.
- `reclaimable_placement_wait` identifies a busy managed lane that can later
  make room. It is distinct from generic live-capacity wait and permanent
  refusal.
- Request retention has fixed per-entry and aggregate byte limits. Discovery
  publishes the capability and limits without prompt content.

## Tests-first evidence

The new integration assertions initially failed at the missing
`queue_position` field. This confirmed that the old timeout response could not
identify a caller's ordered admission state. The same tests then exercised
body-free resume, exact replay, cancellation, expiry, conflict, overlapping
reconnect, and the reclaimable-placement reason.

## Verification

`bash tests/test-negotiator.sh` passed on 2026-09-03. It reported:

- negotiator self-test: PASS
- negotiator integration: PASS
- negotiator pool integration: PASS, including resumable logical admission,
  completed-response replay, cancellation, and terminal admission failures

The suite uses fixture processes and HTTP mocks. It did not load a model, use
CUDA, install the broker, or restart the service.

## Activation boundary

The installed host broker does not receive this protocol until an operator
installs the updated script and restarts the service. Activation is outside
this verification boundary.
