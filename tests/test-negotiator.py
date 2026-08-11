#!/usr/bin/env python3
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


class Backend(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.models = []
        self.requests = []
        self.lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/api/ps":
            self._json({"error": "not found"}, 404)
            return
        with self.server.lock:
            models = list(self.server.models)
        self._json({"models": models})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        with self.server.lock:
            self.server.requests.append((self.path, payload))
            if self.path == "/api/generate":
                if payload.get("keep_alive") == 0:
                    self.server.models = []
                    reason = "unload"
                else:
                    self.server.models = [
                        {
                            "name": payload.get("model", "fixture"),
                            "size": 4096,
                            "size_vram": 2048,
                        }
                    ]
                    reason = "stop"
            else:
                reason = "stop"
        self._json({"done": True, "done_reason": reason})

    def log_message(self, _fmt, *_args):
        pass


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def control(path, payload):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(path)
    client.sendall(json.dumps(payload).encode() + b"\n")
    chunks = []
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    client.close()
    response = json.loads(b"".join(chunks))
    assert response["ok"], response
    return response


def proxy_generate(port, context=262144):
    body = json.dumps(
        {
            "model": "fixture",
            "stream": False,
            "options": {"num_gpu": 999, "main_gpu": 2, "num_ctx": context},
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    response.read()
    conn.close()
    assert response.status == 200, response.status


def well_known(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/.well-known/ollama-unify-gpu-negotiator")
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == 200, response.status
    return payload


def main():
    helper = os.path.abspath(sys.argv[1])
    fixture_bin = os.path.abspath(sys.argv[2])
    backend = Backend()
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy_port = free_port()

    with tempfile.TemporaryDirectory(prefix="ollama-unify-negotiator-") as temp_dir:
        socket_path = os.path.join(temp_dir, "control.sock")
        env = os.environ.copy()
        env.update(
            {
                "PATH": fixture_bin + os.pathsep + env.get("PATH", ""),
                "MOCK_PROFILE": "cuda_triple",
                "OLLAMA_UNIFY_BACKEND": f"127.0.0.1:{backend.server_port}",
                "OLLAMA_UNIFY_LISTEN": f"127.0.0.1:{proxy_port}",
                "OLLAMA_UNIFY_SOCKET": socket_path,
                "OLLAMA_UNIFY_BACKEND_TYPE": "cuda",
                "OLLAMA_UNIFY_SELECTED_GPUS": "GPU-large-0,GPU-large-1,GPU-large-2",
                "OLLAMA_UNIFY_MAX_CONTEXT": "8192",
                "OLLAMA_UNIFY_DRAIN_TIMEOUT": "3",
                "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "3",
                "OLLAMA_UNIFY_LEASE_TTL": "30",
                "OLLAMA_UNIFY_ANON_POLL": "0.1",
                "OLLAMA_UNIFY_ANON_SETTLE": "0.1",
                "OLLAMA_UNIFY_ANON_MAX_DRAIN": "0.5",
            }
        )
        discovery = subprocess.run(
            [helper, "discover"], env=env, check=True, capture_output=True, text=True
        )
        discovery = json.loads(discovery.stdout)
        assert discovery["schema"] == "io.ollama-unify.gpu-negotiator.discovery.v1"
        assert discovery["selected_gpu_count"] == 3
        assert discovery["commands"]["discover"] == "docker gpu discover"
        instructions = subprocess.run(
            [helper, "agent-instructions"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "docker gpu discover" in instructions
        assert "physical GPUs" in instructions
        daemon = subprocess.Popen(
            [helper, "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if daemon.poll() is not None:
                    raise RuntimeError(daemon.stderr.read())
                if os.path.exists(socket_path):
                    try:
                        conn = socket.create_connection(
                            ("127.0.0.1", proxy_port), timeout=0.2
                        )
                        conn.close()
                        break
                    except OSError:
                        pass
                time.sleep(0.05)
            else:
                raise RuntimeError("negotiator did not become ready")

            proxy_generate(proxy_port)
            with backend.lock:
                request = backend.requests[-1][1]
            assert request["options"]["num_gpu"] == -1
            assert "main_gpu" not in request["options"]
            assert request["options"]["num_ctx"] == 8192

            acquired = subprocess.run(
                [
                    helper,
                    "acquire",
                    "--owner",
                    "fixture",
                    "--vram-mib",
                    "4096",
                    "--ttl",
                    "30",
                    "--token-only",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            token = acquired.stdout.strip()
            pending_status = control(socket_path, {"action": "status"})
            assert pending_status["leases"][0]["token"] == token
            assert pending_status["leases"][0]["state"] == "pending"
            with backend.lock:
                assert backend.models == []
            pending_discovery = well_known(proxy_port)
            assert pending_discovery["selected_gpu_count"] == 3

            blocked_error = []
            blocked = threading.Thread(
                target=lambda: proxy_generate(proxy_port, 4096),
                daemon=True,
            )
            blocked.start()
            time.sleep(0.25)
            assert blocked.is_alive(), "proxy request was not held during pending lease"
            ready = control(socket_path, {"action": "ready", "token": token})
            assert ready["lease"]["state"] == "active"
            blocked.join(5)
            assert not blocked.is_alive(), (
                "proxy request did not resume after lease readiness"
            )
            assert not blocked_error

            prepared = control(socket_path, {"action": "prepare", "token": token})
            assert prepared["lease"]["state"] == "pending"
            control(socket_path, {"action": "ready", "token": token})
            released = control(socket_path, {"action": "release", "token": token})
            assert released["released"] == token
            status = control(socket_path, {"action": "status"})
            assert status["leases"] == []
            assert status["draining"] is False
            assert len(status["gpus"]) == 3
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
            if daemon.returncode not in (0, -15):
                raise RuntimeError(daemon.stderr.read())
            backend.shutdown()
            backend.server_close()

    print("negotiator integration: PASS (proxy clamp, drain, lease, resize, release)")


if __name__ == "__main__":
    main()
