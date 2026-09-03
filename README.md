# ollama-unify

> Classify the host, select the safest available Ollama backend, contain memory pressure, and consolidate scattered model stores — without assuming one vendor or one machine shape.

## The problem

Ollama silently ends up with multiple disconnected model libraries on most boxes:

- `~/.ollama/models` — where `ollama serve` from your shell defaults
- `/usr/share/ollama/.ollama/models` — where `ollama.service` (systemd, as user `ollama`) defaults
- `/srv/ollama/models`, `/var/lib/ollama/...` — wherever `OLLAMA_MODELS=` was set in `/etc/default/ollama` or a systemd drop-in
- Whatever your container, agent runtime, or eval harness happens to set when it spawns `ollama serve`

The result: `ollama list` on port 11434 shows a different set of models than `ollama list` on port 11436. Disk fills up with duplicate blobs. Cold-load thrashes the wrong drive. The systemd daemon "runs fine" but serves nothing because all the manifests live in your home directory.

## What it does

`ollama-unify` first classifies the operating system, architecture, effective RAM, cgroup/container boundary, CPU, service manager, installed model payloads, recent Ollama memory projections, and every accelerator backend it can interrogate. It then chooses CUDA, ROCm, Vulkan, Metal, or CPU according to the capabilities actually present and derives scheduler, device-memory, and host-memory limits from those measurements.

The model-store workflow remains independent: it scans every place Ollama stores models, shows exactly what is where, and interactively migrates everything into one canonical location. On compatible systemd GPU hosts, the classified policy also installs a local lease negotiator in front of Ollama. It drains and unloads Ollama before a cooperative external CUDA workload loads, then lets Ollama refit against the VRAM that actually remains. The containment policy refuses unsafe starts, gives Ollama low CPU/I/O priority, asks `systemd-oomd` to kill it under sustained memory pressure, and leaves it stopped after a failure instead of replaying a destructive request loop.

```
  ___  _ _                                       _  __
 / _ \| | | __ _ _ __ ___   __ _    _   _ _ __ (_)/ _|_   _
| | | | | |/ _` | '_ ` _ \ / _` |  | | | | '_ \| | |_| | | |
| |_| | | | (_| | | | | | | (_| |  | |_| | | | | |  _| |_| |
 \___/|_|_|\__,_|_| |_| |_|\__,_|   \__,_|_| |_|_|_|  \__, |
                                                      |___/

Scanning for ollama model stores…

  #   Path                                          Size     Filesystem             Manifests Blobs     Owner
  ──  ────                                          ────     ──────────             ───────── ─────     ─────
  [1] /home/you/.ollama/models                      183G     /dev/nvme0n1p2 (/)     37        74        you:you
  [2] /srv/ollama/models                            73G      /dev/nvme1n1p1 (/srv)  0         12        ollama:ollama

Detected daemons:
  • ollama.service active (PID 696536)
  • 7 manual 'ollama serve' processes (PIDs: 100362, 422930, ...)

Host classification
  Platform: Ubuntu 24.04 LTS (Linux/x86_64; none)
  CPU: 16-Core Workstation CPU — 32 logical cores
  Memory: 65536 MiB (/proc/meminfo)
  Class: workstation; service manager: systemd 255

Accelerator classification
  [cuda/dedicated] GPU 0: NVIDIA 24GB Accelerator, 24576 MiB, compute 8.6, GPU-… (dedicated)
  [cuda/constrained] GPU 1: NVIDIA 2GB Display Adapter, 2048 MiB, compute 6.1 — below 4096 MiB safety floor

Selected Ollama safety policy
  Backend: cuda (dedicated) — highest-confidence native NVIDIA backend
  Device memory: live free-VRAM telemetry; no guessed fixed carve-out
  Model scan: largest installed inference payload 14336 MiB (/srv/ollama/models/manifests/...)
  Ollama history: largest observed host projection 6144 MiB
  Host memory: 2048 MiB startup headroom; throttle at 6144 MiB; hard cap at 14336 MiB; 51200 MiB remains outside the cgroup
  GPU policy: maximum safe GPU layers; spread=0; pageable/cgroup-bounded CPU overflow only
  GPU host paths: unified spill and pinned-host buffers disabled
  Scheduler: 1 model, 1 parallel request, 8192-token context, queue 64

Available mount points for unified storage:
  Filesystem      Size  Used Avail Use% Mounted on
  /dev/nvme0n1p2 1.8T  1.5T  232G  87% /
  /dev/nvme1n1p1 1.8T  100G  1.7T   6% /srv

Where to unify all models? [/srv/ollama/models]: ↵

Update ollama.service via drop-in to use OLLAMA_MODELS=/srv/ollama/models? [Y]: ↵
Apply this classified backend and OOM safety policy to ollama.service? [Y]: ↵
Service runs as 'ollama'. Change to 'you' for single-user simplification? [Y]: ↵
Replace each original store path with a symlink to /srv/ollama/models? [Y]: ↵
Add 'export OLLAMA_MODELS=/srv/ollama/models' to /home/you/.bashrc? [Y]: ↵

Plan
  Destination: /srv/ollama/models
    /home/you/.ollama/models  →  /srv/ollama/models   (183G, strategy: rsync)
    /srv/ollama/models        →  (orphan blobs, archiving)
  Total to move: 183G
  • Point ollama.service at the unified model store
  • Install classified backend/device/host-memory OOM guardrails
  • Change service User/Group to you
  • Symlink originals → destination
  • Add OLLAMA_MODELS export to /home/you/.bashrc
  • Stop running daemons, perform transfer, restart ollama.service

Proceed? [N]: y
```

## Features

- **Discovery** — finds every store referenced by shell env, `/etc/default/ollama`, `/etc/environment`, systemd unit + drop-ins, and live `ollama runner` cmdlines
- **Interactive destination picker** — shows free space per mount, suggests the best candidate
- **Host classifier** — reports OS/architecture, CPU/core count, physical and effective RAM, container/VM status, service manager, and memory-risk tier
- **Backend classifier** — inventories and selects CUDA, ROCm, Vulkan, Metal, or CPU without embedding host-specific UUIDs, GPU counts, or RAM totals
- **Hardware-aware transfer** — picks the fastest supported method automatically:
  - **same filesystem** → `mv` (instant, just a rename)
  - **reflink-capable** (btrfs / XFS / ZFS) → `cp --reflink=auto` (instant CoW)
  - **cross-drive** → local-copy-tuned `rsync`, enabling optimization flags only when that installed rsync supports them
- **Content-addressed dedup** — blobs are SHA-256-named, so `rsync --ignore-existing` collapses duplicates across stores for free
- **Systemd integration** — installs a drop-in to set `OLLAMA_MODELS` and (optionally) change the service `User/Group` to your user for cleaner single-user setups
- **Role-aware accelerator selection** — prefers dedicated devices; display/shared GPUs remain usable when they are the only eligible accelerator
- **Generic capability floors** — classifies constrained or legacy devices from memory and compute telemetry rather than special-casing product names
- **Backend isolation and preflight** — emits the correct visibility variables for the selected backend and validates CUDA, ROCm, or Vulkan telemetry before systemd starts Ollama
- **GPU-first bounded-overflow mode** — delegates placement to Ollama's live-VRAM scheduler and fits the maximum safe number of layers into available VRAM without forcing every selected device into every load; overflow uses ordinary pageable memory inside the service cgroup, never CUDA unified-memory spill, registered host mappings, pinned-host buffers, or swap
- **Live device capacity** — lets Ollama schedule against current free-VRAM telemetry instead of subtracting a guessed percentage; an explicit `OLLAMA_SAFE_VRAM_RESERVE_MIB` remains available when another workload needs a fixed carve-out
- **Dynamic external-GPU negotiation** — installs a streaming API proxy and local lease broker; cooperative workloads load first, after which Ollama automatically refits into the remaining VRAM and moves the unmatched layers to cgroup-bounded pageable RAM
- **Agent and Docker auto-discovery** — registers `docker gpu` as a Docker CLI plugin, publishes a machine-readable discovery manifest and well-known HTTP endpoint, and adds an idempotent managed CUDA policy to the invoking user's existing global Codex instructions
- **Reactive anonymous-process yielding** — detects changes in non-Ollama CUDA process identities, drains Ollama, waits for foreign usage to settle, and reopens it for a fresh live-VRAM fit; this is best-effort because an undeclared process can fail its first allocation before userspace observes it
- **Upgrade-safe pinning** — detects when an Ollama upgrade clears the pinned loopback backend and would drop the daemon back onto the negotiator's public port; `--check-update`, `--reconcile`, and `--update-ollama` repair the drift and cycle the proxy and backend in the only safe order, and an installed path-unit watchdog does the same for upgrades taken outside this script
- **Measured host OOM containment** — sets `MemoryHigh` from the largest recent Ollama host-memory projection and `MemoryMax` from the larger of that projection or the largest installed inference payload; the unallocated host RAM is the result, not a target selected by the script
- **Pressure-aware fail-closed startup** — installs a systemd condition that skips startup without marking the unit failed when `MemAvailable` cannot cover the empty API daemon's bounded startup headroom or memory PSI is already unsafe; model lanes retain live host-memory admission and cgroup containment
- **Proactive pressure killing** — configures `systemd-oomd` to kill Ollama at sustained memory pressure or swap exhaustion before the host becomes unusable
- **CPU and storage protection** — caps Ollama at four logical CPU cores by default and gives its CPU and I/O cgroups low weight, keeping interactive work responsive during cold loads
- **Bounded scheduling** — defaults to one loaded model and one parallel request on every hardware shape, derives context and queue limits from effective RAM, and uses Flash Attention plus `q8_0` KV cache
- **Backward-compat symlinks** — replaces each original path with a symlink to the unified store, so hardcoded references in agents, scripts, and `OLLAMA_MODELS=` overrides keep working
- **Shell rc update** — appends `export OLLAMA_MODELS=…` to your `.bashrc` / `.zshrc` / `config.fish` so new shells use the canonical path natively
- **Safety first** — never deletes data. Originals are renamed to `.bak` (or `.orphan-blobs` for stores with no manifests). You reclaim the space manually when you've confirmed everything works.

## Quick start

```bash
# inspect first (read-only — no daemons stopped, no changes made until you confirm)
curl -sSL https://raw.githubusercontent.com/robit-man/ollama-unify/main/ollama-unify.sh -o ollama-unify.sh
chmod +x ollama-unify.sh
./ollama-unify.sh
```

Or clone:

```bash
git clone https://github.com/robit-man/ollama-unify.git
cd ollama-unify
./ollama-unify.sh
```

Classify a machine without requiring migration tools, sudo, a running Ollama service, or any changes:

```bash
./ollama-unify.sh --classify
```

Preview the generated backend and OOM policy:

```bash
./ollama-unify.sh --safety-preview
```

Install or refresh only the systemd safety policy without scanning or migrating model stores:

```bash
./ollama-unify.sh --install-safety
```

On a dedicated CUDA/ROCm systemd host, that command also installs and boot-enables `ollama-unify-negotiator.service` and the `ollama-unify-gpu-lease` client. Ollama moves to the loopback backend on port `11436`; the negotiator preserves the previously configured public address on port `11434`.

### Dynamic GPU leases

Deployment agents can discover the broker without knowing this repository:

```bash
docker --help                  # lists: gpu*  Negotiate CUDA VRAM with Ollama
docker gpu discover           # machine-readable host policy and selected GPUs
docker gpu status             # live leases, models, VRAM, and cgroup RAM
```

For a long-running Docker ASR/TTS stack, launch it through the broker and use a readiness check that succeeds only after its CUDA models are resident:

```bash
docker gpu run \
  --owner asr-tts \
  --vram-mib 8192 \
  --gpu GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee \
  --ready-command 'curl -fsS http://127.0.0.1:8080/health/ready' \
  -- docker compose up
```

`--gpu` creates a scope that is exclusive against other external leases. While that reservation is pending or revoking, broker-owned Ollama lanes can continue only on other selected GPUs. After the workload calls `ready`, its CUDA footprint is stable, so the broker can place managed Ollama lanes in measured free VRAM on the scoped GPUs with the configured reserve margin. The workload must call `prepare` before later VRAM growth; that transition drains the affected capacity first. The wrapper gives the child exactly the scoped UUIDs through `CUDA_VISIBLE_DEVICES`, heartbeats its lease, and releases it after the foreground command stops. The declared MiB value validates that the selected devices have enough live free memory; it is not an enforceable fractional VRAM carve-out. Without an explicit GPU scope, the broker retains the conservative host-wide drain for compatibility because it cannot prove where the external process will allocate.

For workloads managed by another supervisor, use the explicit lifecycle:

1. `token=$(docker gpu acquire --owner <name> --gpu <uuid> --token-only)` — reserves the specified whole GPU. Add `--gpu` again for each additional device.
2. Start the external workload with `CUDA_VISIBLE_DEVICES` set to exactly those UUIDs, then wait until its CUDA models are fully loaded.
3. `docker gpu ready <token>` — marks the external allocation active and stable. Scoped GPUs become eligible for live-free-VRAM Ollama placement; legacy unscoped leases reopen Ollama here.
4. Before increasing the workload's VRAM use, call `prepare <token>`, resize it, then call `ready <token>` again.
5. Stop the external workload, ensuring its CUDA allocation is gone, then call `release <token>` so Ollama can reload and expand.

To preserve a running workload while upgrading a legacy host-wide lease, use `docker gpu scope <token> --gpu <uuid> [--gpu <uuid> ...]`. The broker accepts the live transition only when all foreign CUDA growth since acquire is contained in the requested external-owner-exclusive scope and the scope can satisfy the original reservation. It then reopens measured free VRAM for broker-owned lanes without stopping the external process. Later foreign growth on an unreserved GPU triggers the existing reactive safety guard.

Use `docker gpu status` to see leases, drain state, loaded Ollama models, foreign CUDA processes, per-GPU memory, and Ollama cgroup memory. The original `ollama-unify-gpu-lease` command remains available when Docker CLI discovery is not applicable. `num_gpu` in the Ollama API means GPU-offloaded model layers—not the number of physical GPUs. The script keeps every selected accelerator visible; on a three-A100 host Ollama may dynamically use one, two, or all three.

### Broker-owned parallel Ollama lanes

Local clients that need concurrent model processes can ask the public broker to create capacity before starting their work:

```bash
curl -fsS http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator/capacity \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:35b","parallel":3}'
```

An embedding client can state its endpoint explicitly when it prewarms a lane by adding `"endpoint":"/api/embed"`. The broker validates this field. It also infers embedding-only and reranking-only warm-up contracts from local model capabilities when the field is absent. Lazy capacity for an ordinary inference request always uses that request's endpoint family. Discovery metadata publishes the reserved private port range so cooperating clients never start a competing Ollama process on a broker lane.

The broker looks up the installed model size and capabilities through `/api/tags`, adds configurable model and VRAM margins, checks each selected GPU independently against live free memory, reserves host headroom, and admits the request only if every missing lane fits. Each accepted lane is a broker-owned `ollama serve` process pinned to one selected GPU and a private loopback port. Placement spreads lanes across available GPUs first, then safely co-locates additional processes on the same GPU only while the conservative live-VRAM budget still fits. The broker loads the requested model through its native completion, embedding, or reranking endpoint and verifies residency before it marks the lane ready. The response exposes only safe lane IDs/GPU assignments and the public API; clients continue sending all inference to port `11434` and cannot select or bypass a private lane. If a client skips the capacity call, its first ordinary inference request lazily starts one fitting lane and concurrent requests expand the pool only while additional lane reservations still fit. Model tags that omit `:latest` share the same lane identity; separately named aliases remain distinct.

Inference uses one broker-owned queue and one capacity reconciler. FIFO order is strict within each model. The broker queue has a hard configurable ceiling (`OLLAMA_UNIFY_POOL_MAX_QUEUE`, default 64); excess callers receive an immediate retryable 503 instead of expanding memory and admission work without bound. A request for another model can bypass a blocked queue head only when its own warm lane is already available, so a large cold load cannot strand usable capacity. The default process ceiling is twice the selected GPU count, while every new lane remains hard-gated by current free VRAM and host-memory reserves. Multiple managed processes can therefore share one GPU when they fit. When every selected GPU is held by a pending or revoking scoped lease, the reconciler parks on lease-state changes instead of retrying an impossible lane placement every two seconds. When a new model does not fit, the broker reclaims the least-recently-used idle lane that has no queued demand, without disturbing active or demanded lanes. Disconnected waiters are removed before they can create phantom inference. Permanent admission failures are returned immediately and removed from the queue, so one incompatible or missing model cannot starve later requests. Runtime status and the well-known endpoint report queue depth, ceiling, age, per-model demand, phases, logical resumes, stale removals, duplicate attempts, and aggregate admission/cancellation/failure timing. Responses return `X-Ollama-Unify-Lane`, `X-Ollama-Unify-Request-Id`, exact admission-only `X-Ollama-Unify-Queue-Ms`, and `X-Ollama-Unify-Queue-Position` headers. Idle lanes are unloaded and stopped automatically, including embedding-only models that reject the generation endpoint. Stable new or increased foreign CUDA allocations affect only broker lanes on selected GPUs; process departure and unselected-GPU churn do not destroy the pool. A scoped cooperative lease temporarily drains the pool while its reservation is committed. Pending and revoking scopes block their GPUs. Active scopes admit Ollama lanes against measured free VRAM because later external growth requires a new `prepare` transition. A legacy unscoped lease keeps the host-wide drain. Explicit release compares aggregate per-GPU foreign use with the pre-lease baseline and allows a small telemetry tolerance (`OLLAMA_UNIFY_FOREIGN_RELEASE_TOLERANCE_MIB`, default 256 MiB) so harmless driver/runtime accounting drift cannot pin an otherwise released scope forever. Broker shutdown also terminates its complete child process groups, and lanes are intentionally ephemeral across restart.

Clients can label admission with `X-Ollama-Unify-Logical-Request-Id`, `X-Ollama-Unify-Workload-Class` (`foreground`, `interactive-control`, or `background`), and `X-Ollama-Unify-Queue-Policy` (`wait` or `yield`). `X-Ollama-Unify-Admission-Wait-Ms` bounds one HTTP admission attempt; it does not limit model loading or generation after admission. With the default `wait` policy and a logical request ID, an expired bounded attempt leaves one detached waiter in its original FIFO position for `OLLAMA_UNIFY_POOL_RESUME_TTL` seconds (default 30). Retrying the identical request with the same logical ID resumes that waiter without another enqueue. A changed body, model, or path under the same ID returns `logical_request_conflict`; an overlapping attempt returns `logical_request_in_progress`; disconnected clients and expired detached waiters are removed. The `yield` policy removes the waiter at the admission deadline.

Admission failures return an explicit `reason_code`, boolean `retryable`, JSON `retry_after_ms` when retryable, and matching `Retry-After`, `X-Ollama-Unify-Retryable`, and `X-Ollama-Unify-Reason-Code` headers. Current reason codes distinguish `queue_admission_timeout`, `queue_full`, `lease_transition`, `lane_capacity_wait`, `host_memory_unavailable`, `model_exceeds_gpu_capacity`, `model_not_installed`, `backend_start_failed`, and invalid or conflicting request contracts. A queue deadline can also include `cause_reason_code`, which preserves the capacity condition observed by the reconciler. Permanent placement failures do not carry `Retry-After`.

A pending lease transition also has an absolute `OLLAMA_UNIFY_PENDING_TIMEOUT` deadline, which defaults to 300 seconds and is not extended by heartbeats. At the deadline, the broker changes the lease to `revoking` and rejects `ready` and `heartbeat`. A scoped lease keeps only its reserved GPUs unavailable, so inference continues on unreserved devices. A legacy unscoped lease fails queued inference with an explicit `503` and keeps Ollama globally drained. The external supervisor must stop the CUDA workload and call `release`; only a verified return to the pre-lease foreign-GPU baseline removes the reservation.

The same discovery document is installed at `/usr/local/share/ollama-unify/gpu-negotiator.json` and served at `/.well-known/ollama-unify-gpu-negotiator` on the public Ollama address. Human-readable cross-agent instructions are installed at `/usr/local/share/ollama-unify/AGENTS.md`. If the invoking account already has `~/.codex`, the installer maintains a marked block in `~/.codex/AGENTS.md`; set `OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY=0` to opt out without disabling Docker or machine-readable discovery.

### Surviving Ollama upgrades

The official Ollama installer rewrites `/etc/systemd/system/ollama.service` and restarts the daemon. The unit it writes carries no `OLLAMA_HOST`, so the pinned loopback backend exists only in the late-priority ollama-unify drop-in. If that drop-in is cleared or drifts, Ollama falls back to its built-in `0.0.0.0:11434` and collides head-on with the negotiator that already owns that address — whichever process loses the bind race dies.

Three commands close that gap:

```bash
./ollama-unify.sh --check-update     # installed vs latest release, plus any pinning drift (read-only)
./ollama-unify.sh --reconcile        # repin and restart the pair after an out-of-band upgrade
./ollama-unify.sh --update-ollama    # update inside a safe stop → repin → start envelope
```

`--update-ollama` stops the negotiator first so the public address is free while the installer runs, lets the official installer do its work, reapplies the pinned safety policy, then restarts the pair in the only safe order — proxy down, backend up, proxy up — and verifies `/api/tags` end to end before reporting success. If the installer fails, the previous policy is reapplied and the stack is brought back up.

`--install-safety` also arms a watchdog for upgrades taken outside this script:

- `/usr/local/libexec/ollama-unify-reconcile` — a standalone repair helper with no dependency on this repository. It reads the pinned addresses from `/usr/local/share/ollama-unify/state.env`, re-asserts `OLLAMA_HOST` in the drop-in and `OLLAMA_UNIFY_BACKEND` in the negotiator config, and cycles the pair in order. If the drop-in has vanished entirely it restores the backend pin alone and logs that `--install-safety` is needed to rebuild the full containment policy.
- `ollama-unify-reconcile.path` — a systemd path unit watching the Ollama binary. Any upgrade that rewrites it triggers the helper, which no-ops when there is no drift.

Print shell-compatible exports for a manual server or a non-systemd supervisor:

```bash
./ollama-unify.sh --print-env
# Review first; then, for the current shell only:
eval "$(./ollama-unify.sh --print-env)"
```

## Requirements

| Required | Optional |
| --- | --- |
| `bash` (4+) and `awk` for classification | `rsync`, `find`, `stat`, `du`, and `df` for model migration |
| `python3` for the generated streaming GPU negotiator on systemd GPU hosts | `curl` for external-workload readiness checks and post-migration API verification |
| | `sudo` (only if writing to protected paths or touching systemd) |
| | `systemctl` (required for service integration and cgroup OOM containment) |
| | `nvidia-smi` for CUDA telemetry |
| | `amd-smi` + `jq`, or `rocminfo`, for ROCm telemetry |
| | `vulkaninfo` for Vulkan telemetry |
| | `system_profiler` for macOS Metal classification |

The fixture suite exercises CUDA dedicated/display/constrained devices, multi-GPU ROCm, integrated Vulkan, Apple Metal, CPU-only hosts, and the negotiator's streaming proxy/drain/lease/resize/release lifecycle. Classification is designed to degrade cleanly on Linux, macOS, FreeBSD, WSL, VMs, and containers. Native Windows requires WSL because this is a Bash utility. systemd cgroup containment and automatic negotiator installation remain Linux-specific.

## Safety guarantees

- **No data is ever deleted.** Source stores are renamed (`<path>.bak` for stores with manifests, `<path>.orphan-blobs` for stores with only orphan blobs). You reclaim the space yourself with `rm -rf` after verifying.
- **The migration flow stops daemons only after you confirm** the plan summary. `--install-safety` is an explicit maintenance command and may briefly restart an active Ollama service so the new boundary takes effect.
- **Idempotent re-runs.** Running the script a second time on a unified setup detects "only one store," reclassifies the host, and can update the systemd safety profile without moving data.
- **Sudo is only requested if needed.** If your destination is in your home dir, no daemons are running, and you skip the systemd step, the script runs with zero privilege escalation.
- **On supported systemd versions, ordinary userspace OOMs are scoped to Ollama.** `MemoryHigh` throttles and reclaims first; `systemd-oomd` reacts to sustained PSI; `MemoryMax` is the last line of defense. `OOMPolicy=stop`, `KillMode=control-group`, and the default `Restart=on-success` keep OOM/driver failures fail-closed while recovering an unexpected clean daemon exit. Explicit `systemctl stop` remains stopped.

## What the script does, step by step

1. Scans the filesystem and current environment for ollama model directories
2. Inspects each one: size, filesystem, owner, manifest/blob counts
3. Detects active daemons (`ollama.service` + manual `ollama serve` processes)
4. Prompts you to pick a unified destination (default = the largest store)
5. Classifies the host and all detectable CUDA, ROCm, Vulkan, Metal, and CPU candidates
6. Selects a backend and derives device, scheduler, and host-memory limits
7. Asks opt-in questions for the systemd model path, OOM guardrails, symlinks, shell rc, and service user
8. Shows the full plan and waits for your final confirmation
9. Stops all daemons cleanly
10. Transfers each non-destination store using the optimal strategy for that source→dest fs pair
11. Archives originals as `.bak` (or `.orphan-blobs`)
12. Normalizes ownership and permissions on the unified destination
13. Installs the memory-pressure preflight and a late-priority systemd drop-in (if requested)
14. Creates backward-compat symlinks (if requested)
15. Appends the env export to your shell rc (if requested)
16. Restarts `ollama.service`
17. Verifies via manifest count and `/api/tags`
18. Prints a backup summary with reclamation commands

## Hardware classification and OOM policy

Heavy inference can exhaust discrete VRAM, shared/unified memory, pinned host pages, swap, or all of them together. The classifier does not assume that every GPU is a dedicated CUDA card—or that a GPU exists at all.

### Automatic backend order

| Candidate | Detection and classification | Selection policy |
| --- | --- |
| CUDA | `nvidia-smi`: UUID, VRAM, display role, compute capability | Dedicated devices first; display devices only when no dedicated CUDA device qualifies; default floor 4 GiB and compute 5.x |
| ROCm | `amd-smi --json` or `rocminfo`: UUID/ordinal, product, VRAM when available | Native AMD candidate; distinguishes discrete, shared/constrained, and unknown-memory devices |
| Vulkan | `vulkaninfo --summary`: device index, name, discrete/integrated/virtual type | Discrete first, then shared/integrated; uses conservative memory estimates because standard summary telemetry lacks free VRAM |
| Metal | macOS `system_profiler` and Apple Silicon architecture | Preferred automatically on macOS; GPU and CPU share unified memory |
| CPU | OS CPU and effective-memory telemetry | Always-available fallback with GPU backends disabled where the platform supports those controls |

On macOS, `auto` prefers Metal. Elsewhere it chooses CUDA, then ROCm, then Vulkan, then CPU. Override an available backend with `OLLAMA_SAFE_BACKEND=cuda|rocm|vulkan|metal|cpu`.

### Dynamic limits

| Guardrail | Derived behavior |
| --- | --- |
| Effective RAM | Physical RAM reduced to the current cgroup limit when running inside a constrained container or service |
| Host class | `<8 GiB` constrained, `<32 GiB` personal, `<128 GiB` workstation, otherwise memory-rich server |
| Installed payload | Sums inference-bearing model/projector/adapter/tensor layers per manifest and selects the largest installed manifest footprint |
| Host throttle | Largest `projected to use … MiB of host memory` value found in the last 30 days of `ollama.service` journal history; falls back to the installed-payload boundary when no projection exists |
| Host hard cap | Larger of the largest installed inference payload and largest observed host projection; the script fails closed instead of inventing a cap when neither a model nor an explicit operator limit exists |
| Host remainder | Effective RAM minus the hard cap. This is an informational outside-cgroup budget; it is not incorrectly treated as memory the empty API daemon must allocate at startup |
| Device capacity | Current free VRAM as measured by Ollama at load time; no automatic fixed carve-out is subtracted a second time |
| Dedicated multi-GPU | Forced spreading is disabled so Ollama can choose devices from live free-VRAM state and split only when needed; aggregate capacity is reported directly from the hardware scan |
| Physical GPU count | Every selected dedicated accelerator remains in the visibility list. API `num_gpu=-1` selects the number of offloaded layers automatically; it does not reduce the host to one physical GPU |
| External lease | Ollama drains and unloads, the external workload allocates first, then Ollama reloads against remaining VRAM; release repeats the cycle so Ollama can expand |
| Dedicated GPU load | Automatic maximum GPU layers, layer splitting and runner fitting enabled, unified-memory spill absent, pinned-host allocation disabled, cgroup-bounded pageable CPU overflow allowed, and cgroup swap disabled |
| Loaded models | 1 on every hardware shape by default; an override is required to allow simultaneous resident models |
| Parallel requests | 1 per model |
| Context | 2,048 below 8 GiB RAM; 4,096 below 16 GiB; 8,192 otherwise (CPU hosts below 32 GiB never exceed 4,096 by default) |
| Request queue | 8, 16, or 64 according to effective RAM |
| Idle retention | 5 minutes rather than indefinitely |
| KV cache | Flash Attention plus `q8_0`, reducing cache growth compared with `f16` |
| Start preflight | Refuses startup when `MemAvailable` is below 2 GiB of bounded daemon headroom (configurable and capped by `MemoryMax`) or at/above 20% full memory PSI over 10 seconds; each model lane separately checks live host memory before load |
| CPU and I/O | 400% CPU quota (capped to host capacity), CPU weight 10, I/O weight 10, and nice level 10 |
| systemd containment | Version-gated `MemoryHigh`, `MemoryMax`, `MemorySwapMax`, `OOMPolicy`, `ManagedOOMMemoryPressure`, and `ManagedOOMSwap` |
| Failure behavior | `Restart=on-success` by default: unexpected clean daemon exits recover, while an OOM, driver failure, or refused preflight remains stopped until the cause is resolved and `systemctl start ollama` is run |

These defaults are intentionally fail-closed. Ollama documents that parallel processing multiplies context allocation, so 32K context × 3 parallel requests can require roughly 96K tokens of context memory per loaded model. Ollama recommends UUIDs for NVIDIA and AMD selection because numeric ordering may vary. llama.cpp enables unified memory by the presence of `GGML_CUDA_ENABLE_UNIFIED_MEMORY`; the generated policy therefore removes it instead of assigning `0`. See Ollama's [concurrency and memory guidance](https://docs.ollama.com/faq), [GPU/backend guidance](https://docs.ollama.com/gpu), llama.cpp's [CUDA build/runtime guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#unified-memory), and AMD's [ROCm isolation guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html).

The defaults can be changed for a single run:

```bash
OLLAMA_SAFE_CONTEXT_LENGTH=16384 \
OLLAMA_SAFE_MAX_LOADED_MODELS=1 \
OLLAMA_SAFE_HOST_RESERVE_MIB=98304 \
OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT=20 \
OLLAMA_SAFE_CPU_QUOTA_PERCENT=400 \
OLLAMA_SAFE_BACKEND=cuda \
./ollama-unify.sh
```

Supported overrides are `OLLAMA_SAFE_BACKEND`, `OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB`, `OLLAMA_SAFE_MIN_GPU_MEMORY_MIB`, `OLLAMA_SAFE_MIN_COMPUTE_MAJOR`, `OLLAMA_SAFE_MODEL_STORE`, `OLLAMA_SAFE_LARGEST_MODEL_MIB`, `OLLAMA_SAFE_OBSERVED_HOST_MIB`, `OLLAMA_SAFE_VRAM_RESERVE_MIB`, `OLLAMA_SAFE_HOST_RESERVE_MIB`, `OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB`, `OLLAMA_SAFE_HOST_MEMORY_MAX_MIB`, `OLLAMA_SAFE_STARTUP_HEADROOM_MIB`, `OLLAMA_SAFE_CONTEXT_LENGTH`, `OLLAMA_SAFE_NUM_PARALLEL`, `OLLAMA_SAFE_MAX_LOADED_MODELS`, `OLLAMA_SAFE_MAX_QUEUE`, `OLLAMA_SAFE_KEEP_ALIVE`, `OLLAMA_SAFE_SWAP_MAX`, `OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT`, `OLLAMA_SAFE_CPU_QUOTA_PERCENT`, `OLLAMA_SAFE_CPU_WEIGHT`, `OLLAMA_SAFE_IO_WEIGHT`, `OLLAMA_SAFE_RESTART_POLICY` (`no`, `on-success`, or `on-failure`), `OLLAMA_SAFE_NEGOTIATOR_LISTEN`, `OLLAMA_SAFE_NEGOTIATOR_GROUP`, `OLLAMA_SAFE_NEGOTIATOR_DRAIN_TIMEOUT`, `OLLAMA_SAFE_NEGOTIATOR_PENDING_TIMEOUT`, `OLLAMA_SAFE_NEGOTIATOR_UNLOAD_TIMEOUT`, `OLLAMA_SAFE_NEGOTIATOR_LEASE_TTL`, `OLLAMA_SAFE_NEGOTIATOR_ANON_POLL`, `OLLAMA_SAFE_NEGOTIATOR_ANON_SETTLE`, `OLLAMA_SAFE_NEGOTIATOR_ANON_MAX_DRAIN`, `OLLAMA_SAFE_POOL_ENABLED`, `OLLAMA_SAFE_POOL_MAX_SERVERS`, `OLLAMA_SAFE_POOL_PORT_START`, `OLLAMA_SAFE_POOL_INSTANCE_PARALLEL`, `OLLAMA_SAFE_POOL_RESUME_TTL`, `OLLAMA_SAFE_POOL_IDLE_TIMEOUT`, `OLLAMA_SAFE_POOL_READY_TIMEOUT`, `OLLAMA_SAFE_POOL_LOAD_TIMEOUT`, `OLLAMA_SAFE_POOL_VRAM_RESERVE_MIB`, `OLLAMA_SAFE_POOL_HOST_RESERVE_MIB`, `OLLAMA_SAFE_POOL_MODEL_OVERHEAD_PERCENT`, `OLLAMA_SAFE_POOL_OLLAMA_BINARY`, and `OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY` (`0` or `1`).

`OLLAMA_CONTEXT_LENGTH` is normally only a server default. When the negotiator proxy is installed, native Ollama API requests are clamped to the context ceiling derived by the hardware scan, and positive `num_gpu`/`main_gpu` overrides are replaced with automatic live fitting. Clients that bypass the proxy and contact the loopback backend directly can bypass those request-level checks; the one-model scheduler, PSI kill, hard cgroup boundary, and failure-only stop policy remain the final containment layer. Under launchd, rc.d, WSL without systemd, or a manually launched server, classification and policy generation still work, but the generated environment must be integrated manually and native cgroup containment is unavailable.

## Common scenarios

**"My shell sees one set of models, my systemd daemon sees another."**

That's the canonical case. The script's default flow handles it.

**"Which backend will this machine use?"**

Run `./ollama-unify.sh --classify`. It reports every candidate, why a device is dedicated/shared/constrained, the selected backend, effective-memory tier, and derived limits. `--safety-preview` also prints the exact policy; `--print-env` emits shell-compatible exports for a manual server or another supervisor.

**"My manual `ollama serve` is much slower than the systemd one."**

That commonly means the manual process inherited different backend visibility or library paths and fell back to CPU. Prefer the managed service so its generated policy applies. On CUDA, if you must launch manually, select inference GPUs by UUID rather than exposing every numeric device:

```bash
export CUDA_VISIBLE_DEVICES=GPU-uuid-for-compute-device-1,GPU-uuid-for-compute-device-2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Discover UUIDs with `nvidia-smi -L`. Do not include a low-memory desktop adapter merely because it appears in `nvidia-smi`; mixed display/inference visibility can make discovery and allocation less predictable.

**"My models are already unified; I only want the crash protection."**

Run the script normally. A single detected store no longer exits immediately: the script skips migration and offers the dynamic systemd safety profile. Use `--safety-preview` first if you only want to inspect the generated policy.

**"I want models on a dedicated drive."**

Mount your fast drive somewhere (e.g., `/srv` or `/data`), pre-create the directory, and pass that path at the "Where to unify all models?" prompt. Cross-filesystem transfers use the fastest locally supported rsync options.

**"I have a container/Docker setup."**

The script handles host-level stores. If Ollama runs in a container, bind-mount the unified path into the container after migration:

```yaml
# docker-compose.yml
volumes:
  - /srv/ollama/models:/root/.ollama/models
```

## Limitations

- **Classification is broader than service integration.** CUDA, ROCm, Vulkan, Metal, CPU, containers, WSL, macOS, and FreeBSD are classified best-effort; automatic cgroup installation currently targets `ollama.service` on systemd.
- **Native Windows needs WSL.** The project is Bash-based and does not install a Windows service policy.
- **ROCm and Vulkan telemetry varies by driver generation.** Missing memory telemetry is reported explicitly instead of being presented as an exact capacity.
- **Single host only.** No remote/multi-host orchestration. For distributed Ollama deployments, run the script per host.
- **Doesn't handle in-flight model pulls.** If `ollama pull` is mid-download when daemons stop, restart it after migration.
- **Systemd containment only covers `ollama.service`.** Manually launched `ollama serve` processes do not inherit the cgroup limits or generated environment.
- **Anonymous CUDA allocation is reactive, not atomic.** The negotiator can detect a new non-Ollama CUDA process and refit after its first successful allocation, but it cannot know an undeclared future `cudaMalloc`. Workloads requiring an OOM guarantee must acquire/resize a lease or run in a fixed hardware partition such as MIG.
- **No userspace policy can guarantee recovery from every driver, firmware, PSU, or hardware failure.** These controls reduce oversubscription and contain normal service OOMs; persistent kernel or accelerator-driver warnings still require platform investigation.

Run the regression suites with `./tests/test-classifier.sh`, `./tests/test-transfer.sh`, and `./tests/test-negotiator.sh`.

## Contributing

Issues and PRs welcome at <https://github.com/robit-man/ollama-unify>.

## License

MIT — see [LICENSE](LICENSE).
