#!/usr/bin/env python3
"""Exercise the tray indicator's menu model without a display."""

import importlib.machinery
import importlib.util
import os
import sys

# A deliberately uneven host: differently sized CUDA devices, one with a
# tiny framebuffer, and one that nvidia-smi inventory did not report.
GPU0 = "GPU-large-0"
GPU1 = "GPU-mid-1"
GPU2 = "GPU-large-2"
GPU3 = "GPU-tiny-3"
GPU4 = "GPU-unlisted-4"
INVENTORY = {
    GPU0: {"index": "0", "name": "Mock CUDA 48GB", "bus": "01:00"},
    GPU1: {"index": "1", "name": "Mock CUDA 24GB", "bus": "02:00"},
    GPU2: {"index": "2", "name": "Mock CUDA 48GB", "bus": "03:00"},
    GPU3: {"index": "3", "name": "Mock CUDA 2GB", "bus": "04:00"},
}
DEVICES = [
    (GPU0, 49152, 30720), (GPU1, 24576, 12288), (GPU2, 49152, 1024),
    (GPU3, 2048, 249), (GPU4, 16384, 0),
]
NOW = 1_800_000_000.0


def load_tray(path):
    loader = importlib.machinery.SourceFileLoader("ollama_unify_tray", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def lease(owner, state, gpu_uuids, created_at, **extra):
    return {
        "token": f"lease_{owner}", "owner": owner, "state": state,
        "requested_mib": 98304, "created_at": created_at,
        "heartbeat_at": NOW - 5, "transition_started_at": created_at,
        "ttl": 300, "foreign_baseline": {}, "gpu_uuids": gpu_uuids,
        "justification": "fixture justification", **extra,
    }


def summary(item, remaining):
    return {
        "owner": item["owner"], "state": item["state"],
        "gpu_uuids": item["gpu_uuids"], "created_at": item["created_at"],
        "justification": item["justification"],
        "seconds_until_expected_release": remaining,
        "horizon_status": "overdue" if remaining < 0 else "expected",
    }


def status_fixture():
    peer = lease("tensor-parallel", "active", [GPU0, GPU2, GPU4], NOW - 3600)
    pending = lease("single-gpu", "pending", [GPU1], NOW - 60)
    revoking = lease("revoked-owner", "revoking", [], NOW - 30)
    return {
        "ok": True, "draining": False, "backend_available": True,
        "leases": [peer, pending, revoking],
        "lease_summaries": [
            summary(peer, 7200), summary(pending, -120), summary(revoking, 600),
        ],
        "gpus": [
            {"uuid": gpu, "total_mib": total, "used_mib": used,
             "free_mib": total - used}
            for gpu, total, used in DEVICES
        ],
        "foreign_gpu_processes": {f"4242@{GPU0}": 30720},
        "clients": [
            {"key": "app:eval", "requests": 12, "first_seen": NOW - 900,
             "last_seen": NOW - 30,
             "identity": {"label": "eval (roctinam)", "declared": "eval",
                          "user": "roctinam", "unit": "matric-tau.service",
                          "pid": 777, "process": "python3",
                          "address": "127.0.0.1", "user_agent": "ollama-python"},
             "models": {"fixture_small:latest": 12},
             "lanes": [
                 {"id": "lane-1", "model": "fixture_small:latest",
                  "gpu_uuid": GPU1, "kind": "managed", "requests": 11},
                 {"id": "lane-0", "model": "fixture_small:latest",
                  "gpu_uuid": GPU0, "kind": "managed", "requests": 1},
                 {"id": "base", "model": "fixture_small:latest",
                  "gpu_uuid": None, "kind": "system", "requests": 0},
             ]},
        ],
        "parallel_pool": {"lanes": [
            {"id": "base", "kind": "system", "gpu_uuid": None},
            {"id": "lane-1", "kind": "managed", "gpu_uuid": GPU1,
             "state": "ready", "model": "fixture_small:latest",
             "parallel": 1, "in_flight": 0, "reserved_mib": 8192,
             "triggered_by": {"key": "app:eval", "label": "eval (roctinam)"},
             "clients": [
                 {"key": "app:eval", "label": "eval (roctinam)",
                  "requests": 12, "last_seen": NOW - 30},
                 {"key": "container:voryn", "label": "container voryn",
                  "requests": 3, "last_seen": NOW - 600},
             ]},
            {"id": "lane-2", "kind": "managed", "gpu_uuid": GPU1,
             "state": "ready", "model": "fixture-busy:latest",
             "parallel": 1, "in_flight": 1, "reserved_mib": 8192},
        ]},
    }


def labels(entry):
    return [action["label"] for action in entry["actions"]]


def requests(entry):
    return {action["label"]: action["request"] for action in entry["actions"]}


def test_menu_model(tray):
    model = tray.build_menu_model(
        status_fixture(), None, INVENTORY, [GPU0, GPU1, GPU2, GPU4],
        {4242: "vllm (pid 4242, docker-abc.scope)"}, NOW,
    )
    assert model["icon"] == tray.ICON_ATTENTION
    assert model["label"] == "3L · 2O"
    assert model["summary"][0] == "Broker running · 3 lease(s) · 2 Ollama lane(s)"

    peer, pending, revoking = model["leases"]
    unlisted = GPU4[:12]
    assert peer["title"] == f"tensor-parallel · GPU0 + GPU2 + {unlisted} · active"
    assert f"GPUs: GPU0 + GPU2 + {unlisted} (exclusive)" in peer["details"]
    assert labels(peer) == [
        "Prepare for resize…", "Renew heartbeat",
        "Remove GPU0 from lease…", "Remove GPU2 from lease…",
        f"Remove {unlisted} from lease…",
        "Revoke lease…", "Release lease…", "Force release…",
    ]
    peer_requests = requests(peer)
    for removed in (GPU0, GPU2, GPU4):
        label = f"Remove {tray.gpu_name(removed, INVENTORY)} from lease…"
        assert peer_requests[label] == {
            "action": "scope", "token": "lease_tensor-parallel",
            "gpu_uuids": [gpu for gpu in (GPU0, GPU2, GPU4) if gpu != removed],
        }
    assert peer_requests["Force release…"]["force"] is True
    assert peer_requests["Revoke lease…"]["action"] == "revoke"
    assert all(action["confirm"] for action in peer["actions"]
               if action["label"] != "Renew heartbeat")

    assert pending["title"] == "single-gpu · GPU1 · pending ⚠"
    assert "Expected release: overdue by 2 min" in pending["details"]
    assert labels(pending)[0] == "Mark ready"
    assert not [label for label in labels(pending) if label.startswith("Remove")]

    assert revoking["title"] == "revoked-owner · all GPUs · revoking ⚠"
    assert labels(revoking) == ["Release lease…", "Force release…"]

    idle, busy = model["lanes"]
    assert idle["title"] == "fixture_small:latest · GPU1 · ready"
    assert "Started for: eval (roctinam)" in idle["details"]
    assert ("Used by: eval (roctinam) · 12 req · under a minute ago"
            in idle["details"])
    assert "Used by: container voryn · 3 req · 10 min ago" in idle["details"]
    assert "Started for: unknown" in busy["details"]

    (client,) = model["clients"]
    assert client["title"] == "eval (roctinam) · 12 req · fixture_small:latest"
    assert client["actions"] == []
    for line in (
        "Declared as: eval", "User: roctinam", "Unit: matric-tau.service",
        "Process: python3 (pid 777)", "User agent: ollama-python",
        "Model: fixture_small:latest · 12 req",
        "Lane lane-1: fixture_small:latest on GPU1 · 11 req · live",
        "Lane lane-0: fixture_small:latest on GPU0 · 1 req · ended",
        "Lane base: fixture_small:latest on base Ollama · 0 req · ended",
    ):
        assert line in client["details"], line
    assert labels(idle) == ["Stop lane…", "Force stop lane…"]
    assert labels(busy) == ["Force stop lane…"]
    assert requests(busy)["Force stop lane…"] == {
        "action": "stop_lane", "lane_id": "lane-2", "force": True,
    }

    gpus = {entry["title"].split(" · ")[0]: entry for entry in model["gpus"]}
    assert gpus["GPU0"]["title"] == "GPU0 · Mock CUDA 48GB · 30 GB / 48 GB · leased"
    assert "Lease: tensor-parallel" in gpus["GPU0"]["details"]
    assert ("CUDA: vllm (pid 4242, docker-abc.scope) · 30 GB"
            in gpus["GPU0"]["details"])
    assert "Ollama: fixture_small:latest (8.0 GB)" in gpus["GPU1"]["details"]
    assert gpus["GPU2"]["title"].endswith("1.0 GB / 48 GB · leased")
    assert gpus["GPU3"]["title"] == (
        "GPU3 · Mock CUDA 2GB · 249 MB / 2.0 GB · not brokered"
    )
    assert gpus[unlisted]["title"] == f"{unlisted} · GPU · 0 MB / 16 GB · leased"
    assert "PCI bus: unknown" in gpus[unlisted]["details"]


def test_single_device_host(tray):
    status = {
        "ok": True, "draining": False, "backend_available": True,
        "leases": [lease("solo", "active", [GPU3], NOW - 10)],
        "lease_summaries": [], "foreign_gpu_processes": {},
        "gpus": [{"uuid": GPU3, "total_mib": 2048, "used_mib": 1900,
                  "free_mib": 148}],
        "parallel_pool": {"lanes": []},
    }
    model = tray.build_menu_model(status, None, {}, [], {}, NOW)
    (solo,) = model["leases"]
    assert "(exclusive)" not in " ".join(solo["details"])
    assert "Expected release: unknown (legacy lease)" in solo["details"]
    assert not [label for label in labels(solo) if label.startswith("Remove")]
    assert model["gpus"][0]["title"] == f"{GPU3[:12]} · GPU · 1.9 GB / 2.0 GB · leased"


def test_units_and_timeouts(tray):
    assert [tray.memory(value) for value in (0, 512, 1536, 10240, 196608)] == [
        "0 MB", "512 MB", "1.5 GB", "10 GB", "192 GB",
    ]
    os.environ.update({
        "OLLAMA_UNIFY_DRAIN_TIMEOUT": "900",
        "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "600",
        "OLLAMA_UNIFY_ANON_MAX_DRAIN": "45",
    })
    try:
        expected = 900 + 600 + 45 + tray.TRANSITION_MARGIN_SECONDS
        assert tray.transition_timeout() == expected
        actions = {
            action["label"]: action
            for action in tray.lease_actions(
                lease("slow", "active", [GPU0], NOW), INVENTORY,
            )
        }
        assert actions["Release lease…"]["timeout"] == expected
        assert actions["Prepare for resize…"]["timeout"] == expected
    finally:
        for name in ("OLLAMA_UNIFY_DRAIN_TIMEOUT", "OLLAMA_UNIFY_UNLOAD_TIMEOUT",
                     "OLLAMA_UNIFY_ANON_MAX_DRAIN"):
            os.environ.pop(name, None)


def row_keys(rows):
    keys = [(row["group"], row["key"]) for row in rows]
    assert len(keys) == len(set(keys)), keys
    for row in rows:
        if row["kind"] == "entry":
            row_keys(row["rows"])
    return [key for _group, key in keys]


def changed_status():
    """Same host a poll later: values moved, one lease left, one lane added."""
    status = status_fixture()
    status["leases"] = [
        item for item in status["leases"] if item["owner"] != "revoked-owner"
    ]
    status["lease_summaries"] = [
        item for item in status["lease_summaries"]
        if item["owner"] != "revoked-owner"
    ]
    status["leases"][0]["heartbeat_at"] = NOW - 600
    status["gpus"][0]["used_mib"] = 45056
    status["parallel_pool"]["lanes"].append({
        "id": "lane-3", "kind": "managed", "gpu_uuid": GPU1,
        "state": "ready", "model": "fixture-new:latest", "parallel": 1,
        "in_flight": 0, "reserved_mib": 4096,
    })
    return status


def build(tray, status):
    return tray.build_menu_model(
        status, None, INVENTORY, [GPU0, GPU1, GPU2, GPU4], {}, NOW,
    )


def test_menu_rows_are_stably_keyed(tray):
    before = tray.menu_rows(build(tray, status_fixture()))
    after = tray.menu_rows(build(tray, changed_status()))
    before_keys, after_keys = row_keys(before), row_keys(after)
    peer = "tensor-parallel@" + str(NOW - 3600)
    assert peer in before_keys and peer in after_keys
    assert "lane-3" in after_keys and "lane-3" not in before_keys
    assert not [key for key in after_keys if "revoked-owner" in key]
    peer_before = next(row for row in before if row["key"] == peer)
    peer_after = next(row for row in after if row["key"] == peer)
    assert [row["key"] for row in peer_before["rows"]] == [
        row["key"] for row in peer_after["rows"]
    ]
    assert peer_before["rows"] != peer_after["rows"]
    groups = []
    for row in before:
        if not groups or groups[-1] != row["group"]:
            groups.append(row["group"])
    assert len(groups) == len(set(groups)), "groups must be contiguous"
    for row in before:
        kinds = {other["kind"] for other in before
                 if other["group"] == row["group"]}
        assert len(kinds) == 1, (row["group"], kinds)


def slot_for(app, key, path=()):
    for group in app.slots.get(path, {}).values():
        for slot in group:
            if slot["row"] is not None and slot["row"]["key"] == key:
                return slot
    raise AssertionError(key)


def widget_count(app):
    return sum(len(pool) for pools in app.slots.values()
               for pool in pools.values())


def test_open_menu_survives_updates(tray):
    """Updates must reuse GTK items; skip without a display."""
    try:
        tray.load_toolkit()
        if not tray.Gtk.init_check(sys.argv)[0]:
            raise RuntimeError("no display")
        app = tray.TrayApp(poll=False)
    except Exception as exc:  # noqa: BLE001 - any toolkit failure means skip
        print(f"tray GTK reconciliation: SKIP ({exc})")
        return
    first, second = build(tray, status_fixture()), build(tray, changed_status())
    app.render(first)
    peer = slot_for(app, "tensor-parallel@" + str(NOW - 3600))
    heartbeat = next(
        slot for pool in app.slots[peer["path"]].values() for slot in pool
        if slot["row"] and slot["row"]["label"].startswith("Heartbeat")
    )
    revoke = slot_for(app, "action:Revoke lease…", peer["path"])
    kept = [(peer, peer["widget"]), (heartbeat, heartbeat["widget"]),
            (revoke, revoke["widget"])]
    gone = slot_for(app, "revoked-owner@" + str(NOW - 30))
    gone_widget = gone["widget"]
    submenu = peer["widget"].get_submenu()

    app.render(second)
    for slot, widget in kept:
        assert slot["widget"] is widget and slot["row"] is not None
    assert peer["widget"].get_submenu() is submenu
    assert heartbeat["widget"].get_label() == "Heartbeat: 10 min ago (TTL 300s)"
    # A vanished entry is hidden in place, never removed, and is not reused
    # by an entry that appeared in the same update.
    assert gone["row"] is None and not gone_widget.get_visible()
    assert gone_widget in app.menu.get_children()
    new_lane = slot_for(app, "lane-3")
    assert new_lane is not gone
    assert new_lane["widget"].get_label() == "fixture-new:latest · GPU1 · ready"
    assert revoke["row"]["spec"]["request"]["token"] == "lease_tensor-parallel"

    # Steady state: once every shape has been seen, alternating data must
    # never create or remove items, i.e. never change the exported layout.
    app.render(first)
    settled = widget_count(app)
    children = list(app.menu.get_children())
    for model in (second, first, second, first):
        app.render(model)
        assert widget_count(app) == settled
        assert list(app.menu.get_children()) == children

    # The broker replaces lanes within one poll (reclaim one, start another)
    # and CUDA processes come and go. Spares absorb both without new items.
    swapped = changed_status()
    lanes = swapped["parallel_pool"]["lanes"]
    lanes[:] = [lane for lane in lanes if lane["id"] != "lane-1"] + [{
        "id": "lane-9", "kind": "managed", "gpu_uuid": GPU2,
        "state": "ready", "model": "fixture-swapped:latest", "parallel": 1,
        "in_flight": 0, "reserved_mib": 2048,
    }]
    swapped["foreign_gpu_processes"][f"5151@{GPU2}"] = 1024
    app.render(build(tray, swapped))
    assert widget_count(app) == settled
    assert list(app.menu.get_children()) == children
    assert slot_for(app, "lane-9")["widget"].get_label() == (
        "fixture-swapped:latest · GPU2 · ready"
    )
    print("tray GTK reconciliation: PASS (items reused and relabelled, "
          "vanished rows hidden in place, entry swaps absorbed by spares, "
          "no layout change in steady state)")


def test_offline_and_idle_models(tray):
    offline = tray.build_menu_model(None, "connection refused", {}, [], {}, NOW)
    assert offline["icon"] == tray.ICON_OFFLINE
    assert offline["summary"] == ["Broker unreachable: connection refused"]

    idle = tray.build_menu_model({
        "ok": True, "draining": False, "backend_available": True,
        "leases": [], "lease_summaries": [], "gpus": [],
        "foreign_gpu_processes": {}, "parallel_pool": {"lanes": []},
    }, None, {}, [], {}, NOW)
    assert idle["icon"] == tray.ICON_OK
    assert idle["leases"] == [] and idle["lanes"] == []


def main():
    tray = load_tray(sys.argv[1])
    test_menu_model(tray)
    test_single_device_host(tray)
    test_units_and_timeouts(tray)
    test_offline_and_idle_models(tray)
    test_menu_rows_are_stably_keyed(tray)
    test_open_menu_survives_updates(tray)
    print(
        "tray indicator model: PASS (heterogeneous devices, N-GPU leases, "
        "per-state actions, lanes, client attribution, units, configured "
        "timeouts, offline, "
        "stable row keys)"
    )


if __name__ == "__main__":
    main()
