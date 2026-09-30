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
        "parallel_pool": {"lanes": [
            {"id": "base", "kind": "system", "gpu_uuid": None},
            {"id": "lane-1", "kind": "managed", "gpu_uuid": GPU1,
             "state": "ready", "model": "fixture_small:latest",
             "parallel": 1, "in_flight": 0, "reserved_mib": 8192},
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
    print(
        "tray indicator model: PASS (heterogeneous devices, N-GPU leases, "
        "per-state actions, lanes, units, configured timeouts, offline)"
    )


if __name__ == "__main__":
    main()
