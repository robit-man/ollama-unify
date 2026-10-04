# WO-06 managed GPU residency attestation

## Objective

Never advertise an Ollama lane as GPU-ready when the model silently loaded on
CPU. Fail closed with a typed terminal error so clients do not wait for a slow
CPU first token, disconnect, destroy the lane, and repeat the cold load.

## Incident anchors

- Telegram router attempts ended at 89 seconds with zero generated tokens.
- The broker repeatedly reported 27B lanes as warm, then killed each lane
  after the client deadline and detached completion TTL.
- The lane's `/api/ps` record reported `size_vram: 0`.
- The zero-token probe showed Ollama launching the 27B model at its 262144-token
  model default even though the broker configured an 8192-token ceiling.
- `nvidia-smi` showed no managed Ollama runner on either selected A100.
- Kernel history reports GPU2 at PCI `84:00.0` fell off the bus (`Xid 79`)
  and the NVIDIA recovery action is `Node Reboot Required` (`Xid 154`).

## Implementation checklist

- [x] Require an exact resident model record after warm-up.
- [x] Require numeric positive `size_vram` for every managed CUDA lane.
- [x] Stop and remove a CPU-fallback lane before it becomes routable.
- [x] Return typed non-retryable `gpu_runtime_unavailable` instead of queueing.
- [x] Publish the new reason code in broker discovery.
- [x] Add a fixture mode that simulates Ollama CPU fallback.
- [x] Add a regression proving zero-VRAM lanes are rejected and unregistered.
- [x] Pin the configured default context during warm-up and when inference
  omits `num_ctx`, so Ollama cannot silently load at the model's larger default.
- [x] Add a regression covering both context-pinning paths.

## Verification and delivery checklist

- [x] Focused CPU-fallback regression passes.
- [x] Full broker integration/static suite passes.
- [x] Repair is committed and pushed to `origin/main`.
- [x] Run `docker gpu discover` immediately before service mutation.
- [x] Install and restart the broker from the verified commit.
- [x] Confirm a zero-token load probe either has positive VRAM residency on
  its scoped healthy GPU or fails immediately with `gpu_runtime_unavailable`.
- [x] Confirm no Telegram router request can enter a CPU-only managed lane.

## Hardware recovery blocker

GPU2 cannot be restored by broker code. The NVIDIA kernel driver reports that
it fell off the PCIe bus and requires a node reboot. Rebooting interrupts the
active Voryn lease and therefore requires an explicit coordinated reboot.

## Verification evidence

- 2026-10-04: full broker, pool, queue, lease, lifecycle, tray, ShellCheck,
  classifier, and transfer suites passed after both repairs.
- 2026-10-04: commits `4860dc8` and `5e40005` were pushed to `origin/main`.
- 2026-10-04: installed negotiator hash matched rendered source; Ollama,
  broker, and tray services were active and the Voryn lease was preserved.
- 2026-10-04: the first GPU0 zero-token probe rejected `size_vram=0` with
  non-retryable `gpu_runtime_unavailable` instead of publishing a lane.
- 2026-10-04: the second probe launched the runner with `-c 8192` rather than
  the erroneous `-c 262144`, then rejected the still-CPU-only load in about
  15 seconds. Queue depth, tracked requests, and managed lanes returned to 0.
- 2026-10-04: `nvidia-smi -q -i 0` reported `GPU Recovery Action: Reboot`;
  kernel history records GPU2 `Xid 79` (fallen off bus) and `Xid 154` (node
  reboot required).
