#!/usr/bin/env python3
import concurrent.futures
import http.client
import http.server
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time


MODEL = "fixture-small:latest"
MODEL_WITHOUT_TAG = "fixture-small"
OTHER_MODEL = "fixture-other:latest"
EMBED_MODEL = "fixture-embed:latest"
REJECT_MODEL = "fixture-reject:latest"
RETRY_MODEL = "fixture-retry:latest"


class Backend(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, tags=None):
        super().__init__(("127.0.0.1", 0), Handler)
        self.models = []
        names = tags if tags is not None else [MODEL]
        self.tags = []
        for item in names:
            if isinstance(item, dict):
                self.tags.append(dict(item))
            else:
                self.tags.append({
                    "name": item,
                    "model": item,
                    "size": 1024**3,
                    "capabilities": ["completion"],
                })
        self.lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, payload, status=200):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with self.server.lock:
            if self.path == "/api/ps":
                payload = {"models": list(self.server.models)}
            elif self.path == "/api/tags":
                payload = {"models": list(self.server.tags)}
            elif self.path == "/api/version":
                payload = {"version": "fixture"}
            else:
                self.send_json({"error": "not found"}, 404)
                return
        self.send_json(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length) or b"{}")
        with self.server.lock:
            if payload.get("keep_alive") == 0:
                self.server.models = []
        self.send_json({"done": True})

    def log_message(self, _fmt, *_args):
        pass


def free_port():
    candidate = socket.socket()
    candidate.bind(("127.0.0.1", 0))
    port = candidate.getsockname()[1]
    candidate.close()
    return port


def http_json(port, method, path, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = json.loads(response.read() or b"{}")
    result = response.status, data, dict(response.getheaders())
    conn.close()
    return result


def control(path, payload):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(path)
    client.sendall(json.dumps(payload).encode() + b"\n")
    response = json.loads(client.makefile("rb").readline())
    client.close()
    assert response["ok"], response
    return response


def events(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []


def wait_ready(daemon, socket_path, proxy_port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if daemon.poll() is not None:
            raise RuntimeError(daemon.stderr.read())
        if os.path.exists(socket_path):
            try:
                connection = socket.create_connection(("127.0.0.1", proxy_port), 0.2)
                connection.close()
                return
            except OSError:
                pass
        time.sleep(0.05)
    raise RuntimeError("pool broker did not become ready")


def wait_until(predicate, description, timeout=5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except (ConnectionError, OSError):
            last = None
        if last:
            return last
        time.sleep(0.025)
    raise AssertionError(f"timed out waiting for {description}; last={last!r}")


def write_compute_apps(path, rows):
    replacement = path + ".new"
    with open(replacement, "w", encoding="utf-8") as stream:
        for pid, gpu_uuid, used_mib in rows:
            stream.write(f"{pid}, {gpu_uuid}, {used_mib}\n")
    os.replace(replacement, path)


def request_events(path):
    return [event for event in events(path)
            if event["kind"] == "request" and event.get("request_id") is not None]


def chat(port, model, request_id, delay=0, timeout=10):
    return http_json(port, "POST", "/api/chat", {
        "model": model,
        "stream": False,
        "mock_request_id": request_id,
        "mock_delay": delay,
    }, timeout=timeout)


def embed(port, model, request_id, timeout=10):
    return http_json(port, "POST", "/api/embed", {
        "model": model,
        "input": "fixture input",
        "mock_request_id": request_id,
    }, timeout=timeout)


def open_abandonable_chat(port, model, request_id):
    payload = json.dumps({
        "model": model,
        "stream": False,
        "mock_request_id": request_id,
    }).encode()
    connection = socket.create_connection(("127.0.0.1", port), timeout=2)
    connection.sendall(
        b"POST /api/chat HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode()
        + b"Connection: close\r\n\r\n"
        + payload
    )
    return connection


def reset_connection(connection):
    connection.setsockopt(
        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
    )
    connection.close()


class PoolHarness:
    def __init__(self, helper, fixture_bin, *, max_servers=1, tags=None,
                 runner_vram_mib=2048):
        self.helper = helper
        self.fixture_bin = fixture_bin
        self.max_servers = max_servers
        self.tags = tags or [MODEL]
        self.runner_vram_mib = runner_vram_mib

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ollama-unify-pool-case-")
        self.temp_dir = self.temp.name
        self.socket_path = os.path.join(self.temp_dir, "control.sock")
        self.event_log = os.path.join(self.temp_dir, "events.jsonl")
        self.compute_apps = os.path.join(self.temp_dir, "compute-apps.csv")
        self.gpu_usage_dir = os.path.join(self.temp_dir, "gpu-usage")
        os.mkdir(self.gpu_usage_dir)
        write_compute_apps(self.compute_apps, [])
        self.backend = Backend(self.tags)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.proxy_port = free_port()
        pool_port = free_port()
        env = os.environ.copy()
        env.update({
            "PATH": self.fixture_bin + os.pathsep + env.get("PATH", ""),
            "MOCK_PROFILE": "cuda_triple",
            "MOCK_OLLAMA_EVENT_LOG": self.event_log,
            "MOCK_NVIDIA_COMPUTE_APPS_FILE": self.compute_apps,
            "MOCK_OLLAMA_GPU_USAGE_DIR": self.gpu_usage_dir,
            "MOCK_OLLAMA_VRAM_MIB": str(self.runner_vram_mib),
            "OLLAMA_UNIFY_BACKEND": f"127.0.0.1:{self.backend.server_port}",
            "OLLAMA_UNIFY_LISTEN": f"127.0.0.1:{self.proxy_port}",
            "OLLAMA_UNIFY_SOCKET": self.socket_path,
            "OLLAMA_UNIFY_LEASE_STATE": os.path.join(self.temp_dir, "leases.json"),
            "OLLAMA_UNIFY_BACKEND_TYPE": "cuda",
            "OLLAMA_UNIFY_SELECTED_GPUS": "GPU-large-0,GPU-large-1,GPU-large-2",
            "OLLAMA_UNIFY_POOL_ENABLED": "1",
            "OLLAMA_UNIFY_POOL_MAX_SERVERS": str(self.max_servers),
            "OLLAMA_UNIFY_POOL_PORT_START": str(pool_port),
            "OLLAMA_UNIFY_POOL_INSTANCE_PARALLEL": "1",
            "OLLAMA_UNIFY_POOL_IDLE_TIMEOUT": "30",
            "OLLAMA_UNIFY_POOL_READY_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_LOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_VRAM_RESERVE_MIB": "1024",
            "OLLAMA_UNIFY_POOL_MODEL_OVERHEAD_PERCENT": "100",
            "OLLAMA_UNIFY_OLLAMA_BINARY": os.path.join(self.fixture_bin, "ollama"),
            "OLLAMA_UNIFY_DRAIN_TIMEOUT": "3",
            "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_ANON_POLL": "0.05",
            "OLLAMA_UNIFY_ANON_SETTLE": "0.1",
            "OLLAMA_UNIFY_ANON_MAX_DRAIN": "0.3",
        })
        self.daemon = subprocess.Popen(
            [self.helper, "serve"], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        wait_ready(self.daemon, self.socket_path, self.proxy_port)
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.daemon.terminate()
        try:
            self.daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.daemon.kill()
            self.daemon.wait()
        self.backend.shutdown()
        self.backend.server_close()
        self.temp.cleanup()

    def status(self):
        return control(self.socket_path, {"action": "status"})

    def capacity(self, model, parallel=1, endpoint=None):
        payload = {"model": model, "parallel": parallel}
        if endpoint is not None:
            payload["endpoint"] = endpoint
        return http_json(
            self.proxy_port, "POST",
            "/.well-known/ollama-unify-gpu-negotiator/capacity",
            payload,
        )


def test_existing_pool_contract(helper, fixture_bin):
    fake_ollama = os.path.join(fixture_bin, "ollama")
    backend = Backend()
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    proxy_port = free_port()
    pool_port = free_port()

    with tempfile.TemporaryDirectory(prefix="ollama-unify-pool-") as temp_dir:
        socket_path = os.path.join(temp_dir, "control.sock")
        event_log = os.path.join(temp_dir, "events.jsonl")
        env = os.environ.copy()
        env.update({
            "PATH": fixture_bin + os.pathsep + env.get("PATH", ""),
            "MOCK_PROFILE": "cuda_triple",
            "MOCK_OLLAMA_EVENT_LOG": event_log,
            "OLLAMA_UNIFY_BACKEND": f"127.0.0.1:{backend.server_port}",
            "OLLAMA_UNIFY_LISTEN": f"127.0.0.1:{proxy_port}",
            "OLLAMA_UNIFY_SOCKET": socket_path,
            "OLLAMA_UNIFY_LEASE_STATE": os.path.join(temp_dir, "leases.json"),
            "OLLAMA_UNIFY_BACKEND_TYPE": "cuda",
            "OLLAMA_UNIFY_SELECTED_GPUS": "GPU-large-0,GPU-large-1,GPU-large-2",
            "OLLAMA_UNIFY_POOL_ENABLED": "1",
            "OLLAMA_UNIFY_POOL_MAX_SERVERS": "3",
            "OLLAMA_UNIFY_POOL_PORT_START": str(pool_port),
            "OLLAMA_UNIFY_POOL_INSTANCE_PARALLEL": "1",
            "OLLAMA_UNIFY_POOL_IDLE_TIMEOUT": "30",
            "OLLAMA_UNIFY_POOL_READY_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_LOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_VRAM_RESERVE_MIB": "1024",
            "OLLAMA_UNIFY_POOL_MODEL_OVERHEAD_PERCENT": "100",
            "OLLAMA_UNIFY_OLLAMA_BINARY": fake_ollama,
            "OLLAMA_UNIFY_DRAIN_TIMEOUT": "3",
            "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_ANON_POLL": "0.1",
        })
        daemon = subprocess.Popen(
            [helper, "serve"], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        try:
            wait_ready(daemon, socket_path, proxy_port)
            discovery_status, discovery, _ = http_json(
                proxy_port, "GET",
                "/.well-known/ollama-unify-gpu-negotiator",
            )
            assert discovery_status == 200, discovery
            assert discovery["parallel_pool"]["private_port_start"] == pool_port
            assert discovery["parallel_pool"]["private_port_end"] == pool_port + 31
            status, capacity, _ = http_json(
                proxy_port, "POST",
                "/.well-known/ollama-unify-gpu-negotiator/capacity",
                {"model": MODEL, "parallel": 2},
            )
            assert status == 200, capacity
            assert capacity["schema"] == "io.ollama-unify.gpu-negotiator.capacity.v1"
            assert capacity["requested_parallel"] == 2
            assert capacity["admitted_parallel"] == 2
            assert capacity["public_ollama_api"] == f"http://127.0.0.1:{proxy_port}"
            assert len(capacity["lanes"]) == 2
            assert len({lane["gpu_uuid"] for lane in capacity["lanes"]}) == 2
            assert all("port" not in lane and "pid" not in lane and "host" not in lane
                       for lane in capacity["lanes"])
            starts = [event for event in events(event_log) if event["kind"] == "start"]
            assert len(starts) == 2
            assert len({event["host"] for event in starts}) == 2
            assert len({event["gpu"] for event in starts}) == 2

            _, repeated, _ = http_json(
                proxy_port, "POST",
                "/.well-known/ollama-unify-gpu-negotiator/capacity",
                {"model": MODEL, "parallel": 2},
            )
            assert repeated["admitted_parallel"] == 2
            assert len([event for event in events(event_log)
                        if event["kind"] == "start"]) == 2

            def chat(request_id):
                return http_json(proxy_port, "POST", "/api/chat", {
                    "model": MODEL, "stream": False, "mock_request_id": request_id,
                    "mock_delay": 0.5,
                })

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(chat, index) for index in (1, 2)]
                results = [future.result(timeout=5) for future in futures]
            assert all(result[0] == 200 for result in results)
            lane_headers = [result[2]["X-Ollama-Unify-Lane"] for result in results]
            assert len(set(lane_headers)) == 2
            ps_status, running, _ = http_json(proxy_port, "GET", "/api/ps")
            assert ps_status == 200
            assert len(running["models"]) == 2
            assert len({model["ollama_unify_lane"] for model in running["models"]}) == 2

            before = len([event for event in events(event_log) if event["kind"] == "start"])
            status, rejected, _ = http_json(
                proxy_port, "POST",
                "/.well-known/ollama-unify-gpu-negotiator/capacity",
                {"model": MODEL, "parallel": 4},
            )
            assert status == 503, rejected
            assert len([event for event in events(event_log)
                        if event["kind"] == "start"]) == before

            acquired = control(socket_path, {
                "action": "acquire", "owner": "pool-fixture",
                "requested_mib": 1024, "ttl": 30,
            })
            assert len(acquired["stopped_lanes"]) == 2
            assert len([event for event in events(event_log)
                        if event["kind"] == "stop"]) == 2
            token = acquired["lease"]["token"]
            remaining_lanes = control(socket_path, {"action": "status"})[
                "parallel_pool"
            ]["lanes"]
            assert len(remaining_lanes) == 1
            assert remaining_lanes[0]["kind"] == "system"
            metadata_started = time.monotonic()
            tags_status, tags, _ = http_json(proxy_port, "GET", "/api/tags")
            assert tags_status == 200
            assert tags["models"][0]["name"] == MODEL
            pending_ps_status, pending_models, _ = http_json(
                proxy_port, "GET", "/api/ps"
            )
            assert pending_ps_status == 200
            assert pending_models == {"models": []}
            assert time.monotonic() - metadata_started < 1
            control(socket_path, {"action": "ready", "token": token})
            control(socket_path, {"action": "release", "token": token})

            lazy_status, _, lazy_headers = chat(3)
            assert lazy_status == 200
            assert lazy_headers["X-Ollama-Unify-Lane"].startswith("lane-")
            assert len([event for event in events(event_log)
                        if event["kind"] == "start"]) == before + 1
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
            observed = events(event_log)
            started_pids = {event["pid"] for event in observed if event["kind"] == "start"}
            stopped_pids = {event["pid"] for event in observed if event["kind"] == "stop"}
            assert started_pids <= stopped_pids
            backend.shutdown()
            backend.server_close()


def test_scoped_pending_lease_preserves_unreserved_inference(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=3, tags=[MODEL, OTHER_MODEL],
    ) as harness:
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "scoped-fixture",
            "requested_mib": 1024,
            "ttl": 30,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1"],
        })
        token = acquired["lease"]["token"]
        assert acquired["lease"]["gpu_uuids"] == [
            "GPU-large-0", "GPU-large-1",
        ]
        pending = harness.status()
        assert pending["draining"] is False
        assert pending["leases"][0]["state"] == "pending"

        status, payload, headers = chat(
            harness.proxy_port, MODEL, "during-scoped-pending", timeout=10
        )
        assert status == 200, payload
        assert headers["X-Ollama-Unify-Lane"].startswith("lane-")
        managed = managed_lanes(harness.status())
        assert len(managed) == 1
        assert managed[0]["gpu_uuid"] == "GPU-large-2"
        starts = [event for event in events(harness.event_log)
                  if event["kind"] == "start"]
        assert starts
        assert all(event["gpu"] == "GPU-large-2" for event in starts)

        status, payload, headers = chat(
            harness.proxy_port, OTHER_MODEL, "replace-on-only-free-gpu", timeout=10
        )
        assert status == 200, payload
        assert headers["X-Ollama-Unify-Lane"].startswith("lane-")
        replacement = managed_lanes(harness.status())
        assert len(replacement) == 2
        assert {lane["model"] for lane in replacement} == {MODEL, OTHER_MODEL}
        assert {lane["gpu_uuid"] for lane in replacement} == {"GPU-large-2"}
        lifecycle = events(harness.event_log)
        assert not any(event["kind"] == "stop" for event in lifecycle)
        assert [
            event["gpu"] for event in lifecycle if event["kind"] == "start"
        ] == ["GPU-large-2", "GPU-large-2"]

        control(harness.socket_path, {"action": "ready", "token": token})
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token
        assert harness.status()["draining"] is False


def test_scoped_release_tolerates_baseline_pid_churn_without_global_drain(
    helper, fixture_bin,
):
    with PoolHarness(helper, fixture_bin, max_servers=2, tags=[MODEL]) as harness:
        # A long-running external service can replace a worker PID while its
        # stable CUDA allocation remains identical. The lease baseline must be
        # compared by scoped GPU usage, not by process identity.
        write_compute_apps(harness.compute_apps, [
            (910101, "GPU-large-0", 1024),
        ])
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "scoped-pid-churn-fixture",
            "requested_mib": 4096,
            "ttl": 30,
            "gpu_uuids": ["GPU-large-0"],
        })
        token = acquired["lease"]["token"]
        control(harness.socket_path, {"action": "ready", "token": token})

        capacity_status, capacity, _ = harness.capacity(MODEL)
        assert capacity_status == 200, capacity
        lane_ids = [lane["id"] for lane in managed_lanes(harness.status())]
        assert lane_ids

        write_compute_apps(harness.compute_apps, [
            (910202, "GPU-large-0", 1024),
        ])
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token
        assert released["stopped_lanes"] == []
        status = harness.status()
        assert status["draining"] is False
        assert status["leases"] == []
        assert [lane["id"] for lane in managed_lanes(status)] == lane_ids


def test_scoped_active_lease_shares_stable_vram_with_ollama(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL],
    ) as harness:
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "active-shared-fixture",
            "requested_mib": 98304,
            "ttl": 30,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1", "GPU-large-2"],
        })
        token = acquired["lease"]["token"]

        pending_status, pending_payload, _ = harness.capacity(MODEL)
        assert pending_status == 503, pending_payload
        assert "free=[]" in pending_payload["error"]

        ready = control(harness.socket_path, {"action": "ready", "token": token})
        assert ready["lease"]["state"] == "active"

        active_status, active_capacity, _ = harness.capacity(MODEL)
        assert active_status == 200, active_capacity
        assert active_capacity["admitted_parallel"] == 1
        assert len(active_capacity["lanes"]) == 1
        assert active_capacity["lanes"][0]["gpu_uuid"] in {
            "GPU-large-0", "GPU-large-1", "GPU-large-2",
        }

        response = chat(
            harness.proxy_port, MODEL, "during-active-scoped-lease", timeout=10
        )
        assert response[0] == 200, response[1]
        assert response[2]["X-Ollama-Unify-Lane"].startswith("lane-")

        prepared = control(
            harness.socket_path, {"action": "prepare", "token": token}
        )
        assert prepared["lease"]["state"] == "pending"
        assert prepared["stopped_lanes"]
        assert managed_lanes(harness.status()) == []

        control(harness.socket_path, {"action": "ready", "token": token})
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token


def test_scoped_acquire_preserves_managed_lanes_when_allocation_fits(
    helper, fixture_bin,
):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL],
    ) as harness:
        initial_status, initial_capacity, _ = harness.capacity(MODEL)
        assert initial_status == 200, initial_capacity
        lane_id = initial_capacity["lanes"][0]["id"]
        starts_before = len([
            event for event in events(harness.event_log)
            if event["kind"] == "start"
        ])

        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "fitting-scoped-fixture",
            "requested_mib": 98304,
            "ttl": 30,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1", "GPU-large-2"],
        })
        token = acquired["lease"]["token"]
        assert acquired["stopped_lanes"] == []
        pending_lanes = managed_lanes(harness.status())
        assert [lane["id"] for lane in pending_lanes] == [lane_id]

        control(harness.socket_path, {"action": "ready", "token": token})
        response = chat(
            harness.proxy_port, MODEL, "reuse-after-scoped-ready", timeout=10
        )
        assert response[0] == 200, response[1]
        assert response[2]["X-Ollama-Unify-Lane"] == lane_id
        assert len([
            event for event in events(harness.event_log)
            if event["kind"] == "start"
        ]) == starts_before

        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token


def capacity_when_ready(harness, model, parallel=1):
    status, payload, headers = harness.capacity(model, parallel)
    return (status, payload, headers) if status == 200 else None


def managed_lanes(status):
    return [lane for lane in status["parallel_pool"]["lanes"]
            if lane["kind"] == "managed"]


def queue_with_depth(harness, depth):
    queue = harness.status()["parallel_pool"]["queue"]
    return queue if queue["depth"] == depth else None


def test_foreign_gpu_transition_stability(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        start = wait_until(
            lambda: next((event for event in events(harness.event_log)
                          if event["kind"] == "start"), None),
            "managed lane start",
        )

        # A managed child must never be classified as an anonymous CUDA process,
        # including in this fixture where no real systemd cgroup is available.
        write_compute_apps(harness.compute_apps, [
            (start["pid"], start["gpu"], 1024),
        ])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]

        # Process churn on an unselected GPU and allocation departure are inert.
        write_compute_apps(harness.compute_apps, [
            (start["pid"], start["gpu"], 1024),
            (910001, "GPU-unselected", 4096),
        ])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]
        write_compute_apps(harness.compute_apps, [])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]

        # A stable new allocation on a selected GPU triggers one rebalance.
        write_compute_apps(harness.compute_apps, [
            (910002, "GPU-large-0", 1024),
        ])
        wait_until(
            lambda: len([event for event in events(harness.event_log)
                         if event["kind"] == "stop"]) >= 1,
            "rebalance after selected-GPU allocation",
        )
        first_stop_count = len([event for event in events(harness.event_log)
                                if event["kind"] == "stop"])
        wait_until(
            lambda: capacity_when_ready(harness, MODEL),
            "capacity recovery after selected-GPU allocation",
        )

        # Growth by the same process is also material and triggers a new fit.
        write_compute_apps(harness.compute_apps, [
            (910002, "GPU-large-0", 2048),
        ])
        wait_until(
            lambda: len([event for event in events(harness.event_log)
                         if event["kind"] == "stop"]) > first_stop_count,
            "rebalance after selected-GPU allocation growth",
        )
        second_stop_count = len([event for event in events(harness.event_log)
                                 if event["kind"] == "stop"])
        wait_until(
            lambda: capacity_when_ready(harness, MODEL),
            "capacity recovery after selected-GPU allocation growth",
        )

        # Releasing foreign VRAM must not destroy the newly fitted lane.
        write_compute_apps(harness.compute_apps, [])
        time.sleep(0.3)
        assert len([event for event in events(harness.event_log)
                    if event["kind"] == "stop"]) == second_stop_count


def test_implicit_latest_uses_one_lane(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL_WITHOUT_TAG)
        assert status == 200, capacity
        assert capacity["canonical_model"] == MODEL
        lane_id = capacity["lanes"][0]["id"]

        explicit = chat(harness.proxy_port, MODEL, "explicit-latest")
        implicit = chat(harness.proxy_port, MODEL_WITHOUT_TAG, "implicit-latest")
        assert explicit[0] == implicit[0] == 200
        assert explicit[2]["X-Ollama-Unify-Lane"] == lane_id
        assert implicit[2]["X-Ollama-Unify-Lane"] == lane_id
        assert len([event for event in events(harness.event_log)
                    if event["kind"] == "start"]) == 1
        assert [event["model"] for event in request_events(harness.event_log)] == [
            MODEL, MODEL_WITHOUT_TAG,
        ]


def test_idle_lane_replacement(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=1, tags=[MODEL, OTHER_MODEL]
    ) as harness:
        status, first_capacity, _ = harness.capacity(MODEL)
        assert status == 200, first_capacity
        first_lane = first_capacity["lanes"][0]["id"]
        first = chat(harness.proxy_port, MODEL, "first-model")
        assert first[0] == 200

        second = chat(harness.proxy_port, OTHER_MODEL, "replacement-model", timeout=8)
        assert second[0] == 200, second
        assert second[2]["X-Ollama-Unify-Lane"] != first_lane
        observed = events(harness.event_log)
        assert len([event for event in observed if event["kind"] == "start"]) == 2
        assert len([event for event in observed if event["kind"] == "stop"]) == 1
        lanes = managed_lanes(harness.status())
        assert len(lanes) == 1
        assert lanes[0]["model"] == OTHER_MODEL


def test_live_vram_reclaims_idle_lane_below_process_ceiling(helper, fixture_bin):
    model_size = 60 * 1024**3
    tags = [
        {"name": MODEL, "model": MODEL, "size": model_size,
         "capabilities": ["completion"]},
        {"name": OTHER_MODEL, "model": OTHER_MODEL, "size": model_size,
         "capabilities": ["completion"]},
    ]
    required_mib = (model_size // (1024 * 1024)) + 1024
    with PoolHarness(
        helper, fixture_bin, max_servers=6, tags=tags,
        runner_vram_mib=required_mib,
    ) as harness:
        status, first_capacity, _ = harness.capacity(MODEL, parallel=3)
        assert status == 200, first_capacity
        assert len(first_capacity["lanes"]) == 3

        status, payload, _ = chat(
            harness.proxy_port, OTHER_MODEL, "vram-reclamation", timeout=10
        )
        assert status == 200, payload
        observed = events(harness.event_log)
        assert len([event for event in observed if event["kind"] == "start"]) == 4
        assert len([event for event in observed if event["kind"] == "stop"]) == 1
        lanes = managed_lanes(harness.status())
        assert len(lanes) == 3
        assert [lane["model"] for lane in lanes].count(MODEL) == 2
        assert [lane["model"] for lane in lanes].count(OTHER_MODEL) == 1


def test_lazy_parallel_scaling(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=3) as harness:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    chat, harness.proxy_port, MODEL, f"lazy-{index}", 0.5, 10
                )
                for index in (1, 2, 3)
            ]
            results = [future.result(timeout=10) for future in futures]
        assert all(result[0] == 200 for result in results)
        assert len({result[2]["X-Ollama-Unify-Lane"] for result in results}) == 3
        assert len([event for event in events(harness.event_log)
                    if event["kind"] == "start"]) == 3
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["enqueued_total"] == 3
        assert queue["admitted_total"] == 3
        assert queue["cancelled_total"] == 0


def test_fifo_queue_and_metrics(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(
                chat, harness.proxy_port, MODEL, "fifo-1", 0.8, 8
            )
            wait_until(
                lambda: any(event.get("request_id") == "fifo-1"
                            for event in request_events(harness.event_log)),
                "first FIFO request admission",
            )
            second = executor.submit(
                chat, harness.proxy_port, MODEL, "fifo-2", 0.05, 8
            )
            first_queue = wait_until(
                lambda: queue_with_depth(harness, 1),
                "one queued FIFO request",
            )
            third = executor.submit(
                chat, harness.proxy_port, MODEL, "fifo-3", 0.05, 8
            )
            queue = wait_until(
                lambda: queue_with_depth(harness, 2),
                "two queued FIFO requests",
            )
            assert queue["by_model"] == {MODEL: 2}
            assert queue["phase_counts"].get("queued") == 2
            assert queue["oldest_wait_ms"] >= first_queue["oldest_wait_ms"]
            assert [item["position"] for item in queue["requests"]] == [1, 2]
            assert all(item["model"] == MODEL for item in queue["requests"])

            results = [future.result(timeout=8) for future in (first, second, third)]

        assert all(result[0] == 200 for result in results)
        assert [event["request_id"] for event in request_events(
            harness.event_log
        )] == ["fifo-1", "fifo-2", "fifo-3"]
        assert queue_with_depth(harness, 0)
        for result in results:
            assert result[2]["X-Ollama-Unify-Request-Id"]
            assert int(result[2]["X-Ollama-Unify-Queue-Ms"]) >= 0
            assert int(result[2]["X-Ollama-Unify-Queue-Position"]) >= 1
        final_queue = harness.status()["parallel_pool"]["queue"]
        assert final_queue["enqueued_total"] == 3
        assert final_queue["admitted_total"] == 3
        assert final_queue["peak"] >= 2
        assert final_queue["wait_ms_max"] >= final_queue["wait_ms_mean"]


def test_ready_model_bypasses_blocked_other_model(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL, OTHER_MODEL]
    ) as harness:
        assert harness.capacity(MODEL)[0] == 200
        assert harness.capacity(OTHER_MODEL)[0] == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            other_active = executor.submit(
                chat, harness.proxy_port, OTHER_MODEL, "other-active", 1.5, 8
            )
            model_active = executor.submit(
                chat, harness.proxy_port, MODEL, "model-active", 0.4, 8
            )
            wait_until(
                lambda: {
                    event.get("request_id")
                    for event in request_events(harness.event_log)
                } >= {"other-active", "model-active"},
                "both model lanes to become active",
            )

            other_waiting = executor.submit(
                chat, harness.proxy_port, OTHER_MODEL, "other-waiting", 0.05, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 1),
                "blocked other-model queue head",
            )
            model_waiting = executor.submit(
                chat, harness.proxy_port, MODEL, "model-bypass", 0.05, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 2),
                "cross-model waiter behind blocked queue head",
            )

            assert model_active.result(timeout=2)[0] == 200
            assert model_waiting.result(timeout=1)[0] == 200
            assert other_active.result(timeout=3)[0] == 200
            assert other_waiting.result(timeout=2)[0] == 200

        request_ids = [
            event["request_id"] for event in request_events(harness.event_log)
        ]
        assert request_ids.index("model-bypass") < request_ids.index("other-waiting")
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]
        final_queue = harness.status()["parallel_pool"]["queue"]
        assert final_queue["depth"] == 0
        assert final_queue["timed_out_total"] == 0


def test_queued_disconnect_is_cancelled(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(
                chat, harness.proxy_port, MODEL, "active-before-cancel", 1.0, 8
            )
            wait_until(
                lambda: any(event.get("request_id") == "active-before-cancel"
                            for event in request_events(harness.event_log)),
                "active request before cancellation",
            )
            abandoned = open_abandonable_chat(
                harness.proxy_port, MODEL, "must-not-run"
            )
            wait_until(
                lambda: queue_with_depth(harness, 1),
                "abandoned request queue admission",
            )
            reset_connection(abandoned)
            wait_until(
                lambda: queue_with_depth(harness, 0),
                "abandoned request cancellation",
            )
            assert active.result(timeout=8)[0] == 200

        time.sleep(0.15)
        assert not any(event.get("request_id") == "must-not-run"
                       for event in request_events(harness.event_log))
        status = harness.status()
        assert status["active_requests"] == 0
        assert status["parallel_pool"]["queue"]["depth"] == 0
        assert status["parallel_pool"]["queue"]["cancelled_total"] == 1
        assert status["parallel_pool"]["queue"]["admitted_total"] == 1
        assert all(lane["in_flight"] == 0 for lane in managed_lanes(status))


def test_embedding_model_uses_embedding_warmup(helper, fixture_bin):
    tags = [{
        "name": EMBED_MODEL,
        "model": EMBED_MODEL,
        "size": 1024**3,
        "capabilities": ["embedding"],
    }]
    with PoolHarness(helper, fixture_bin, max_servers=1, tags=tags) as harness:
        capacity_status, capacity, _ = harness.capacity(
            EMBED_MODEL, endpoint="/api/embed"
        )
        assert capacity_status == 200, capacity
        status, payload, headers = embed(
            harness.proxy_port, EMBED_MODEL, "embedding-request", timeout=8
        )
        assert status == 200, payload
        assert headers["X-Ollama-Unify-Lane"].startswith("lane-")
        observed = [event for event in events(harness.event_log)
                    if event["kind"] == "request"
                    and event["model"] == EMBED_MODEL]
        assert [event["path"] for event in observed] == [
            "/api/embed", "/api/embed",
        ]
        assert observed[-1]["request_id"] == "embedding-request"


def test_permanent_failure_does_not_block_fifo(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=1, tags=[MODEL, REJECT_MODEL]
    ) as harness:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rejected = executor.submit(
                chat, harness.proxy_port, REJECT_MODEL, "rejected-model", 0, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 1),
                "rejected model queue admission",
            )
            accepted = executor.submit(
                chat, harness.proxy_port, MODEL, "after-missing", 0, 8
            )
            rejected_result = rejected.result(timeout=8)
            accepted_result = accepted.result(timeout=8)
        assert rejected_result[0] == 400, rejected_result
        assert "does not support requested operation" in rejected_result[1]["error"]
        assert accepted_result[0] == 200, accepted_result
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["depth"] == 0
        assert queue["rejected_total"] == 1
        assert queue["admitted_total"] == 1
        assert queue["timed_out_total"] == 0


def test_retryable_warm_failure_stays_queued(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1, tags=[RETRY_MODEL]) as harness:
        status, payload, _ = chat(
            harness.proxy_port, RETRY_MODEL, "after-retry", 0, 10
        )
        assert status == 200, payload
        warm_attempts = [
            event for event in events(harness.event_log)
            if event["kind"] == "request"
            and event["model"] == RETRY_MODEL
            and event["request_id"] is None
        ]
        assert len(warm_attempts) == 2, warm_attempts
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["rejected_total"] == 0
        assert queue["admitted_total"] == 1


def test_capacity_rejects_unknown_endpoint(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, payload, _ = harness.capacity(MODEL, endpoint="/api/unknown")
        assert status == 400, payload
        assert "supported Ollama inference path" in payload["error"]


def main():
    helper = os.path.abspath(sys.argv[1])
    fixture_bin = os.path.abspath(sys.argv[2])
    test_existing_pool_contract(helper, fixture_bin)
    test_scoped_pending_lease_preserves_unreserved_inference(helper, fixture_bin)
    test_scoped_release_tolerates_baseline_pid_churn_without_global_drain(
        helper, fixture_bin,
    )
    test_scoped_active_lease_shares_stable_vram_with_ollama(helper, fixture_bin)
    test_scoped_acquire_preserves_managed_lanes_when_allocation_fits(
        helper, fixture_bin,
    )
    test_foreign_gpu_transition_stability(helper, fixture_bin)
    test_implicit_latest_uses_one_lane(helper, fixture_bin)
    test_idle_lane_replacement(helper, fixture_bin)
    test_live_vram_reclaims_idle_lane_below_process_ceiling(helper, fixture_bin)
    test_lazy_parallel_scaling(helper, fixture_bin)
    test_fifo_queue_and_metrics(helper, fixture_bin)
    test_ready_model_bypasses_blocked_other_model(helper, fixture_bin)
    test_queued_disconnect_is_cancelled(helper, fixture_bin)
    test_embedding_model_uses_embedding_warmup(helper, fixture_bin)
    test_permanent_failure_does_not_block_fifo(helper, fixture_bin)
    test_retryable_warm_failure_stays_queued(helper, fixture_bin)
    test_capacity_rejects_unknown_endpoint(helper, fixture_bin)
    print(
        "negotiator pool integration: PASS "
        "(fit, endpoint-aware warm lanes, stable watcher, aliases, replacement, "
        "per-model FIFO, warm-lane bypass, cancellation, terminal admission failures)"
    )


if __name__ == "__main__":
    main()
