# Cooperative external GPU handoff — 2026-10-10

Request: Voryn should automatically recover and gracefully relocate to an
available GPU when another lease requests its current GPU. The companion Voryn
tracker is `docs/deployment/voice-gpu-recovery-progress.md` in robit-man/voryn.

- [x] Inspect installed protocol and active lease summaries without contention.
- [x] Isolate work from the divergent Desktop main checkout in a clean worktree
  based on upstream3de0c379c7b38331169cc79ea118136313e27388.
- [x] Add explicit, persisted single-GPU owner opt-in, default false.
- [x] Add bounded token-free request intent without revoking incumbent ownership.
- [x] Wait outside the transition lock so verified release can complete.
- [x] Guard scope acquisition, scope change and cache evacuation against stealing
  the intended successor's scope, without misclassifying intent as registered CUDA.
- [x] Reject hold-and-wait cycles, invalid metadata and impossible total capacity
  before initiating a yield; preserve non-opted-in and transitioning owners.
- [x] Cancel waiting intent on disconnect, timeout or daemon restart.
- [x] Preserve ordinary health, capacity, pending, readiness and release gates.
- [x] Publish capability and intent through daemon/CLI discovery; extend the
  CLI's acquire transport deadline to include the bounded protocol maximum.
- [x] Author seven CPU control/discovery regression methods; wire the existing
  offline negotiator suite to include them.
- [x] Root review and static Bash/ShellCheck/Python syntax/whitespace checks.
- [ ] Execute regression suites under explicit approval (currently held).
- [ ] Roll out broker and companion supervisor under explicit approval.
- [ ] Qualify real ownership handoff and ASR/TTS recovery; not claimed.

No installed executable, daemon, incident hold or unrelated workload was changed.
This is code delivery only, not live GPU migration qualification. The incumbent
must actually implement the advertised cooperative shutdown contract before it
opts in; the generic foreground runner deliberately does not opt in.
