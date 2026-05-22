# ollama-unify

> Consolidate scattered Ollama model stores into one canonical location — interactively, safely, and at full hardware speed.

## The problem

Ollama silently ends up with multiple disconnected model libraries on most boxes:

- `~/.ollama/models` — where `ollama serve` from your shell defaults
- `/usr/share/ollama/.ollama/models` — where `ollama.service` (systemd, as user `ollama`) defaults
- `/srv/ollama/models`, `/var/lib/ollama/...` — wherever `OLLAMA_MODELS=` was set in `/etc/default/ollama` or a systemd drop-in
- Whatever your container, agent runtime, or eval harness happens to set when it spawns `ollama serve`

The result: `ollama list` on port 11434 shows a different set of models than `ollama list` on port 11436. Disk fills up with duplicate blobs. Cold-load thrashes the wrong drive. The systemd daemon "runs fine" but serves nothing because all the manifests live in your home directory.

## What it does

`ollama-unify` scans every place Ollama stores models, shows you exactly what's where, then interactively migrates everything into a single canonical location of your choosing.

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
Service runs as 'ollama'. Change to 'you' for single-user simplification? [Y]: ↵
Replace each original store path with a symlink to /srv/ollama/models? [Y]: ↵
Add 'export OLLAMA_MODELS=/srv/ollama/models' to /home/you/.bashrc? [Y]: ↵

Plan
  Destination: /srv/ollama/models
    /home/you/.ollama/models  →  /srv/ollama/models   (183G, strategy: rsync)
    /srv/ollama/models        →  (orphan blobs, archiving)
  Total to move: 183G
  • Install systemd drop-in
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

## Requirements

| Required | Optional |
| --- | --- |
| `bash` (4+), `rsync`, `find`, `stat`, `awk` | `sudo` (only if writing to system paths or touching systemd) |
| | `systemctl` (only if you have `ollama.service` installed) |
| | `curl` (used for the post-migration `/api/tags` check) |

Tested on Linux (Ubuntu 22.04+, Debian 12, Fedora 39+). macOS uses launchd rather than systemd — the discovery + transfer phases will work but the systemd steps are skipped automatically.

## Safety guarantees

- **No data is ever deleted.** Source stores are renamed (`<path>.bak` for stores with manifests, `<path>.orphan-blobs` for stores with only orphan blobs). You reclaim the space yourself with `rm -rf` after verifying.
- **Daemons are stopped only after you confirm** the plan summary. Aborting at the confirmation prompt is a no-op.
- **Idempotent re-runs.** Running the script a second time on a unified setup detects "only one store" and exits cleanly.
- **Sudo is only requested if needed.** If your destination is in your home dir, no daemons are running, and you skip the systemd step, the script runs with zero privilege escalation.

## What the script does, step by step

1. Scans the filesystem and current environment for ollama model directories
2. Inspects each one: size, filesystem, owner, manifest/blob counts
3. Detects active daemons (`ollama.service` + manual `ollama serve` processes)
4. Prompts you to pick a unified destination (default = the largest store)
5. Asks four opt-in questions for systemd / symlinks / shell rc / service user
6. Shows the full plan and waits for your final confirmation
7. Stops all daemons cleanly
8. Transfers each non-destination store using the optimal strategy for that source→dest fs pair
9. Archives originals as `.bak` (or `.orphan-blobs`)
10. Normalizes ownership and permissions on the unified destination
11. Installs the systemd drop-in (if requested)
12. Creates backward-compat symlinks (if requested)
13. Appends the env export to your shell rc (if requested)
14. Restarts `ollama.service`
15. Verifies via manifest count and `/api/tags`
16. Prints a backup summary with reclamation commands

## Common scenarios

**"My shell sees one set of models, my systemd daemon sees another."**

That's the canonical case. The script's default flow handles it.

**"My manual `ollama serve` is 50× slower than the systemd one."**

Almost always means your shell is missing `CUDA_VISIBLE_DEVICES` / `LD_LIBRARY_PATH` and the manual daemon silently fell back to CPU. After unification, also add to your shell rc:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

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

## Contributing

Issues and PRs welcome at <https://github.com/robit-man/ollama-unify>.

## License

MIT — see [LICENSE](LICENSE).
