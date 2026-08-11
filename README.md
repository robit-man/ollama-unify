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

`ollama-unify` first classifies the operating system, architecture, effective RAM, cgroup/container boundary, CPU, service manager, and every accelerator backend it can interrogate. It then chooses CUDA, ROCm, Vulkan, Metal, or CPU according to the capabilities actually present and derives conservative scheduler, device-memory, and host-memory limits.

The model-store workflow remains independent: it scans every place Ollama stores models, shows exactly what is where, and interactively migrates everything into one canonical location. On compatible systemd hosts, the classified policy is installed as a late-priority service drop-in. It refuses unsafe starts, gives Ollama low CPU/I/O priority, asks `systemd-oomd` to kill it under sustained memory pressure, and leaves it stopped after a failure instead of replaying a destructive request loop.

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
  Device-memory reserve: 2457 MiB per selected accelerator
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
- **Role-aware accelerator selection** — prefers dedicated devices; display/shared GPUs remain usable when they are the only safe accelerator, with a larger memory reserve
- **Generic capability floors** — classifies constrained or legacy devices from memory and compute telemetry rather than special-casing product names
- **Backend isolation and preflight** — emits the correct visibility variables for the selected backend and validates CUDA, ROCm, or Vulkan telemetry before systemd starts Ollama
- **Dynamic device headroom** — reserves 10% of dedicated or 20% of shared device memory, subject to size-aware floors and a 16 GiB ceiling; conservative estimates are used when a backend cannot expose memory
- **Host OOM containment** — reserves 35% of effective RAM on ordinary hosts or 10% on accelerator-rich multi-GPU servers, derives `MemoryHigh` and `MemoryMax`, limits service swap, and makes Ollama the preferred OOM victim
- **Pressure-aware fail-closed startup** — installs a systemd condition that skips startup without marking the unit failed when `MemAvailable` is below the reserve or memory PSI is already unsafe
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
| | `sudo` (only if writing to protected paths or touching systemd) |
| | `systemctl` (required for service integration and cgroup OOM containment) |
| | `nvidia-smi` for CUDA telemetry |
| | `amd-smi` + `jq`, or `rocminfo`, for ROCm telemetry |
| | `vulkaninfo` for Vulkan telemetry |
| | `system_profiler` for macOS Metal classification |
| | `curl` (used for the post-migration `/api/tags` check) |

The fixture suite exercises CUDA dedicated/display/constrained devices, multi-GPU ROCm, integrated Vulkan, Apple Metal, and CPU-only hosts. Classification is designed to degrade cleanly on Linux, macOS, FreeBSD, WSL, VMs, and containers. Native Windows requires WSL because this is a Bash utility. systemd cgroup containment remains Linux-specific.

## Safety guarantees

- **No data is ever deleted.** Source stores are renamed (`<path>.bak` for stores with manifests, `<path>.orphan-blobs` for stores with only orphan blobs). You reclaim the space yourself with `rm -rf` after verifying.
- **The migration flow stops daemons only after you confirm** the plan summary. `--install-safety` is an explicit maintenance command and may briefly restart an active Ollama service so the new boundary takes effect.
- **Idempotent re-runs.** Running the script a second time on a unified setup detects "only one store," reclassifies the host, and can update the systemd safety profile without moving data.
- **Sudo is only requested if needed.** If your destination is in your home dir, no daemons are running, and you skip the systemd step, the script runs with zero privilege escalation.
- **On supported systemd versions, ordinary userspace OOMs are scoped to Ollama.** `MemoryHigh` throttles and reclaims first; `systemd-oomd` reacts to sustained PSI; `MemoryMax` is the last line of defense. `OOMPolicy=stop`, `KillMode=control-group`, and the default `Restart=no` terminate the entire service cgroup and do not automatically replay the failed workload.

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
| Host reserve | Starts at 35% on ordinary hosts. When the scan finds at least two dedicated GPUs whose summed reported VRAM is at least half of host RAM, it uses 10% and a 16 GiB floor |
| Device reserve | 10% of the smallest dedicated CUDA/ROCm device or 20% for a shared device, with size-aware floors and a 16 GiB ceiling |
| Unknown GPU memory | 2 GiB ROCm estimate or 1 GiB Vulkan estimate |
| Loaded models | 1 on every hardware shape by default; an override is required to allow simultaneous resident models |
| Parallel requests | 1 per model |
| Context | 2,048 below 8 GiB RAM; 4,096 below 16 GiB; 8,192 otherwise (CPU hosts below 32 GiB never exceed 4,096 by default) |
| Request queue | 8, 16, or 64 according to effective RAM |
| Idle retention | 5 minutes rather than indefinitely |
| KV cache | Flash Attention plus `q8_0`, reducing cache growth compared with `f16` |
| Throttle target | Normally the hard cap minus a 10% band. For a scanned accelerator-rich host, the target is the lower of summed dedicated VRAM or the safe host ceiling; no GPU model, count, or host size is hardcoded |
| Start preflight | Refuses startup below the host-memory reserve or at/above 20% full memory PSI over 10 seconds |
| CPU and I/O | 400% CPU quota (capped to host capacity), CPU weight 10, I/O weight 10, and nice level 10 |
| systemd containment | Version-gated `MemoryHigh`, `MemoryMax`, `MemorySwapMax`, `OOMPolicy`, `ManagedOOMMemoryPressure`, and `ManagedOOMSwap` |
| Failure behavior | `Restart=no` by default, so an OOM, driver failure, or refused preflight requires an explicit `systemctl start ollama` after the cause is resolved |

These defaults are intentionally conservative. Ollama documents that parallel processing multiplies context allocation, so 32K context × 3 parallel requests can require roughly 96K tokens of context memory per loaded model. Ollama recommends UUIDs for NVIDIA and AMD selection because numeric ordering may vary. See Ollama's [concurrency and memory guidance](https://docs.ollama.com/faq), [GPU/backend guidance](https://docs.ollama.com/gpu), and AMD's [ROCm isolation guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html).

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

Supported overrides are `OLLAMA_SAFE_BACKEND`, `OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB`, `OLLAMA_SAFE_MIN_GPU_MEMORY_MIB`, `OLLAMA_SAFE_MIN_COMPUTE_MAJOR`, `OLLAMA_SAFE_VRAM_RESERVE_MIB`, `OLLAMA_SAFE_HOST_RESERVE_MIB`, `OLLAMA_SAFE_CONTEXT_LENGTH`, `OLLAMA_SAFE_NUM_PARALLEL`, `OLLAMA_SAFE_MAX_LOADED_MODELS`, `OLLAMA_SAFE_MAX_QUEUE`, `OLLAMA_SAFE_KEEP_ALIVE`, `OLLAMA_SAFE_SWAP_MAX`, `OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT`, `OLLAMA_SAFE_CPU_QUOTA_PERCENT`, `OLLAMA_SAFE_CPU_WEIGHT`, `OLLAMA_SAFE_IO_WEIGHT`, and `OLLAMA_SAFE_RESTART_POLICY` (`no` or `on-failure`).

`OLLAMA_CONTEXT_LENGTH` is a server default, not an API-enforced maximum. A client can still request a larger `num_ctx`, and API `keep_alive` can override model retention. The one-model scheduler, PSI kill, hard cgroup boundary, and no-restart policy are therefore the containment layer: a pathological request can fail Ollama, but it should not be allowed to replay until it wedges the host. Under launchd, rc.d, WSL without systemd, or a manually launched server, classification and policy generation still work, but the generated environment must be integrated with that service manually and native cgroup containment is unavailable.

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
- **ROCm and Vulkan telemetry varies by driver generation.** Missing memory telemetry is reported explicitly and uses a conservative estimate rather than pretending it is exact.
- **Single host only.** No remote/multi-host orchestration. For distributed Ollama deployments, run the script per host.
- **Doesn't handle in-flight model pulls.** If `ollama pull` is mid-download when daemons stop, restart it after migration.
- **Systemd containment only covers `ollama.service`.** Manually launched `ollama serve` processes do not inherit the cgroup limits or generated environment.
- **No userspace policy can guarantee recovery from every driver, firmware, PSU, or hardware failure.** These controls reduce oversubscription and contain normal service OOMs; persistent kernel or accelerator-driver warnings still require platform investigation.

Run the classifier regression suite with `./tests/test-classifier.sh`.

## Contributing

Issues and PRs welcome at <https://github.com/robit-man/ollama-unify>.

## License

MIT — see [LICENSE](LICENSE).
