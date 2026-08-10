# ollama-unify

> Consolidate scattered Ollama model stores into one canonical location, then optionally install hardware-aware GPU and OOM guardrails — interactively, safely, and at full hardware speed.

## The problem

Ollama silently ends up with multiple disconnected model libraries on most boxes:

- `~/.ollama/models` — where `ollama serve` from your shell defaults
- `/usr/share/ollama/.ollama/models` — where `ollama.service` (systemd, as user `ollama`) defaults
- `/srv/ollama/models`, `/var/lib/ollama/...` — wherever `OLLAMA_MODELS=` was set in `/etc/default/ollama` or a systemd drop-in
- Whatever your container, agent runtime, or eval harness happens to set when it spawns `ollama serve`

The result: `ollama list` on port 11434 shows a different set of models than `ollama list` on port 11436. Disk fills up with duplicate blobs. Cold-load thrashes the wrong drive. The systemd daemon "runs fine" but serves nothing because all the manifests live in your home directory.

## What it does

`ollama-unify` scans every place Ollama stores models, shows you exactly what's where, then interactively migrates everything into a single canonical location of your choosing. On systemd hosts with NVIDIA GPUs it can also install a late-priority service drop-in that selects suitable CUDA inference accelerators, reserves VRAM and host RAM dynamically, bounds Ollama concurrency, and contains ordinary service OOMs before they can consume all host memory.

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

Available mount points for unified storage:
  Mounted on    Size    Avail   Type
  /             1.8T    232G    ext4
  /srv          1.8T    1.7T    ext4

Where to unify all models? [/srv/ollama/models]: ↵

Update ollama.service via drop-in to use OLLAMA_MODELS=/srv/ollama/models? [Y]: ↵
Apply these dynamic GPU and OOM safety guardrails to ollama.service? [Y]: ↵
Service runs as 'ollama'. Change to 'you' for single-user simplification? [Y]: ↵
Replace each original store path with a symlink to /srv/ollama/models? [Y]: ↵
Add 'export OLLAMA_MODELS=/srv/ollama/models' to /home/you/.bashrc? [Y]: ↵

Plan
  Destination: /srv/ollama/models
    /home/you/.ollama/models  →  /srv/ollama/models   (183G, strategy: rsync)
    /srv/ollama/models        →  (orphan blobs, archiving)
  Total to move: 183G
  • Point ollama.service at the unified model store
  • Install dynamic GPU/VRAM/host-memory OOM guardrails
  • Change service User/Group to you
  • Symlink originals → destination
  • Add OLLAMA_MODELS export to /home/you/.bashrc
  • Stop running daemons, perform transfer, restart ollama.service

Proceed? [N]: y
```

## Features

- **Discovery** — finds every store referenced by shell env, `/etc/default/ollama`, `/etc/environment`, systemd unit + drop-ins, and live `ollama runner` cmdlines
- **Interactive destination picker** — shows free space per mount, suggests the best candidate
- **Hardware-aware transfer** — picks the fastest method automatically:
  - **same filesystem** → `mv` (instant, just a rename)
  - **reflink-capable** (btrfs / XFS / ZFS) → `cp --reflink=auto` (instant CoW)
  - **cross-drive** → NVMe-tuned `rsync` (`--whole-file --inplace --no-compress --ignore-existing`)
- **Content-addressed dedup** — blobs are SHA-256-named, so `rsync --ignore-existing` collapses duplicates across stores for free
- **Systemd integration** — installs a drop-in to set `OLLAMA_MODELS` and (optionally) change the service `User/Group` to your user for cleaner single-user setups
- **CUDA-only inference GPU selection** — chooses non-display NVIDIA GPUs by UUID, VRAM, and CUDA compute capability; low-memory/display adapters such as the GT 1030 are excluded, while ROCm, Vulkan, and integrated-GPU discovery are disabled
- **Start-time GPU preflight** — checks every selected UUID through `nvidia-smi` before Ollama starts, avoiding a missing-GPU restart silently becoming a CPU-only workload
- **Dynamic VRAM headroom** — reserves 8% of the smallest selected GPU (bounded to 4–16 GiB) so Ollama does not schedule models against every reported byte
- **Host OOM containment** — derives `MemoryHigh` and `MemoryMax` from installed RAM, limits service swap, and makes systemd stop/restart only Ollama if its cgroup reaches the hard boundary
- **Bounded scheduling** — conservative defaults for loaded models, parallel requests, context length, queue depth, and model retention; Flash Attention and `q8_0` KV cache reduce context-memory growth
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

Preview the detected GPUs and generated OOM policy without sudo, stopping Ollama, or changing any files:

```bash
./ollama-unify.sh --safety-preview
```

## Requirements

| Required | Optional |
| --- | --- |
| `bash` (4+), `rsync`, `find`, `stat`, `awk` | `sudo` (only if writing to system paths or touching systemd) |
| | `systemctl` (required for service integration and OOM containment) |
| | `nvidia-smi` (required for dynamic NVIDIA GPU/VRAM guardrails) |
| | `curl` (used for the post-migration `/api/tags` check) |

Tested on Linux (Ubuntu 22.04+, Debian 12, Fedora 39+). macOS uses launchd rather than systemd — the discovery + transfer phases will work but the systemd steps are skipped automatically.

## Safety guarantees

- **No data is ever deleted.** Source stores are renamed (`<path>.bak` for stores with manifests, `<path>.orphan-blobs` for stores with only orphan blobs). You reclaim the space yourself with `rm -rf` after verifying.
- **Daemons are stopped only after you confirm** the plan summary. Aborting at the confirmation prompt is a no-op.
- **Idempotent re-runs.** Running the script a second time on a unified setup detects "only one store" and still offers the systemd safety profile without moving data.
- **Sudo is only requested if needed.** If your destination is in your home dir, no daemons are running, and you skip the systemd step, the script runs with zero privilege escalation.
- **Ordinary userspace OOMs are scoped to Ollama.** `MemoryHigh` throttles and reclaims first; `MemoryMax` is the last line of defense. `OOMPolicy=stop` and `Restart=on-failure` terminate and restart the service cgroup, substantially reducing the chance that pressure escalates into a host-wide OOM.

## What the script does, step by step

1. Scans the filesystem and current environment for ollama model directories
2. Inspects each one: size, filesystem, owner, manifest/blob counts
3. Detects active daemons (`ollama.service` + manual `ollama serve` processes)
4. Prompts you to pick a unified destination (default = the largest store)
5. Shows the dynamic CUDA/VRAM/host-memory safety profile when supported
6. Asks opt-in questions for the systemd model path, OOM guardrails, symlinks, shell rc, and service user
7. Shows the full plan and waits for your final confirmation
8. Stops all daemons cleanly
9. Transfers each non-destination store using the optimal strategy for that source→dest fs pair
10. Archives originals as `.bak` (or `.orphan-blobs`)
11. Normalizes ownership and permissions on the unified destination
12. Installs a late-priority systemd drop-in (if requested)
13. Creates backward-compat symlinks (if requested)
14. Appends the env export to your shell rc (if requested)
15. Restarts `ollama.service`
16. Verifies via manifest count and `/api/tags`
17. Prints a backup summary with reclamation commands

## GPU and OOM safety profile

Heavy multi-GPU model loads can fail in more ways than a normal userspace allocation. A CUDA runner may exhaust VRAM, repeatedly time out during GPU discovery, or ask the NVIDIA driver to pin a large host-memory region. If the driver or kernel becomes unresponsive, the result can be an abrupt machine reset with no useful OOM-killer or panic record.

The optional safety profile addresses the controllable pressure points:

| Guardrail | Default behavior |
| --- | --- |
| GPU eligibility | Display inactive, at least 16 GiB VRAM, CUDA compute capability 7+, and not a GT 1030 |
| GPU identity | Stable GPU UUIDs rather than reorderable numeric indices |
| Backend isolation | CUDA only; ROCm, Vulkan, and integrated-GPU discovery disabled |
| Restart preflight | Every selected UUID must answer an `nvidia-smi` query before the service starts |
| VRAM reserve | 8% of the smallest eligible GPU, clamped to 4–16 GiB per GPU |
| Loaded models | 2 on multi-GPU systems; 1 on a single-GPU system |
| Parallel requests | 1 per model |
| Default context | 8,192 tokens |
| Request queue | 64 requests before Ollama rejects overload |
| Idle model retention | 5 minutes rather than indefinitely |
| KV cache | Flash Attention plus `q8_0` cache (about half the cache memory of `f16`, with a small precision tradeoff) |
| Host reserve | 20% of RAM, with a 64 GiB floor on hosts with at least 128 GiB RAM |
| Hard containment | systemd `MemoryMax`, `MemorySwapMax`, `OOMPolicy`, and bounded restart rate |

These defaults are intentionally conservative. Ollama documents that parallel processing multiplies context allocation, so a configuration such as 32K context × 3 parallel requests can require roughly 96K tokens of context memory per loaded model. Ollama also recommends UUIDs for GPU selection because numeric ordering can vary. See the official [concurrency and memory guidance](https://docs.ollama.com/faq) and [GPU selection guidance](https://docs.ollama.com/gpu).

The defaults can be changed for a single run:

```bash
OLLAMA_SAFE_CONTEXT_LENGTH=16384 \
OLLAMA_SAFE_MAX_LOADED_MODELS=1 \
OLLAMA_SAFE_HOST_RESERVE_MIB=98304 \
./ollama-unify.sh
```

Supported overrides are `OLLAMA_SAFE_MIN_GPU_MEMORY_MIB`, `OLLAMA_SAFE_MIN_COMPUTE_MAJOR`, `OLLAMA_SAFE_VRAM_RESERVE_MIB`, `OLLAMA_SAFE_HOST_RESERVE_MIB`, `OLLAMA_SAFE_CONTEXT_LENGTH`, `OLLAMA_SAFE_NUM_PARALLEL`, `OLLAMA_SAFE_MAX_LOADED_MODELS`, `OLLAMA_SAFE_MAX_QUEUE`, `OLLAMA_SAFE_KEEP_ALIVE`, and `OLLAMA_SAFE_SWAP_MAX`.

`OLLAMA_CONTEXT_LENGTH` is a server default, not an API-enforced maximum. A client can still request a larger `num_ctx`, and API `keep_alive` can override model retention. The systemd hard boundary remains the final containment layer for such requests. The start preflight prevents fallback when an expected GPU is absent or `nvidia-smi` cannot reach the driver; it cannot detect every CUDA runtime incompatibility that might make Ollama fall back to CPU after launch.

## Common scenarios

**"My shell sees one set of models, my systemd daemon sees another."**

That's the canonical case. The script's default flow handles it.

**"My manual `ollama serve` is 50× slower than the systemd one."**

Almost always means your shell is missing `CUDA_VISIBLE_DEVICES` / `LD_LIBRARY_PATH` and the manual daemon silently fell back to CPU. Prefer the managed systemd service so the generated guardrails apply. If you must launch manually, select inference GPUs by UUID rather than exposing every numeric device:

```bash
export CUDA_VISIBLE_DEVICES=GPU-uuid-for-compute-device-1,GPU-uuid-for-compute-device-2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Discover UUIDs with `nvidia-smi -L`. Do not include a low-memory desktop adapter merely because it appears in `nvidia-smi`; mixed display/inference visibility can make discovery and allocation less predictable.

**"My models are already unified; I only want the crash protection."**

Run the script normally. A single detected store no longer exits immediately: the script skips migration and offers the dynamic systemd safety profile. Use `--safety-preview` first if you only want to inspect the generated policy.

**"I want models on a dedicated drive."**

Mount your fast drive somewhere (e.g., `/srv` or `/data`), pre-create the directory, and pass that path at the "Where to unify all models?" prompt. The transfer will use rsync at full NVMe speed.

**"I have a container/Docker setup."**

The script handles host-level stores. If Ollama runs in a container, bind-mount the unified path into the container after migration:

```yaml
# docker-compose.yml
volumes:
  - /srv/ollama/models:/root/.ollama/models
```

## Limitations

- **Linux-first.** macOS users will get useful discovery output but systemd steps are skipped silently. PRs to add launchd plist editing are welcome.
- **Single host only.** No remote/multi-host orchestration. For distributed Ollama deployments, run the script per host.
- **Doesn't handle in-flight model pulls.** If `ollama pull` is mid-download when daemons stop, restart it after migration.
- **Systemd containment only covers `ollama.service`.** Manually launched `ollama serve` processes do not inherit the cgroup limits or generated environment.
- **No userspace policy can guarantee recovery from every driver, firmware, PSU, or hardware failure.** These controls prevent oversubscription and contain normal service OOMs; persistent NVIDIA kernel warnings still require driver/kernel investigation.

## Contributing

Issues and PRs welcome at <https://github.com/robit-man/ollama-unify>.

## License

MIT — see [LICENSE](LICENSE).
