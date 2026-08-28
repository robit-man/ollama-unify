#!/usr/bin/env python3
import concurrent.futures
import http.client
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time


MODEL = "fixture-small:latest"


class Backend(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.models = []
        self.tags = [{"name": MODEL, "model": MODEL, "size": 1024**3}]
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


def main():
    helper = os.path.abspath(sys.argv[1])
    fixture_bin = os.path.abspath(sys.argv[2])
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

    print("negotiator pool integration: PASS (atomic fit, spawn, route, drain, cleanup)")


if __name__ == "__main__":
    main()
