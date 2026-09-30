#!/usr/bin/env python3
"""Exercise client identification helpers without sockets or privileges."""

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import time


def load_negotiator(path):
    # Keep the host's broker configuration out of the imported module.
    os.environ["OLLAMA_UNIFY_CONFIG"] = "/nonexistent/ollama-unify-negotiator"
    loader = importlib.machinery.SourceFileLoader("negotiator", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def test_parse_socket_owner(negotiator):
    line = (
        "0      0      127.0.0.1:41994 127.0.0.1:11434 "
        "timer:(keepalive,50sec,0) uid:1005 ino:3196166 sk:6001 "
        "cgroup:/system.slice/matric-tau.service <->\n"
    )
    assert negotiator.parse_socket_owner(line) == {
        "uid": 1005, "inode": 3196166, "cgroup": "/system.slice/matric-tau.service",
    }
    # Older iproute2 omits cgroup; identification still uses the uid.
    assert negotiator.parse_socket_owner(
        "0 0 [::1]:5000 [::1]:11434 uid:0 ino:12 sk:1 <->"
    ) == {"uid": 0, "inode": 12, "cgroup": ""}
    assert negotiator.parse_socket_owner("") is None


def test_cgroup_unit(negotiator):
    assert negotiator.cgroup_unit("/system.slice/matric-tau.service") == (
        "matric-tau.service", "",
    )
    assert negotiator.cgroup_unit(
        "/system.slice/docker-3b73baae0c53aa.scope"
    ) == ("docker-3b73baae0c53aa.scope", "3b73baae0c53aa")
    assert negotiator.cgroup_unit(
        "/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "app-org.gnome.Terminal.slice/vte-spawn-1.scope"
    ) == ("vte-spawn-1.scope", "")
    assert negotiator.cgroup_unit("/") == ("", "")


def test_keys_and_labels(negotiator):
    key = negotiator.client_key_and_label
    assert key({"declared": "eval-runner", "user": "roctinam",
                "container": "ignored"}) == (
        "app:eval-runner", "eval-runner (roctinam)",
    )
    assert key({"container": "voryn", "address": "172.18.0.3"}) == (
        "container:voryn", "container voryn",
    )
    assert key({"unit": "matric-tau.service", "user": "roctinam",
                "process": "python3"}) == (
        "unit:roctinam:matric-tau.service", "matric-tau.service (roctinam)",
    )
    # Interactive scopes are per terminal; the process names the app.
    assert key({"unit": "vte-spawn-1.scope", "user": "roko",
                "process": "python3"}) == (
        "process:roko:python3", "python3 (roko)",
    )
    assert key({"address": "192.168.1.39", "user_agent": "ollama-js/0.5"}) == (
        "remote:192.168.1.39", "192.168.1.39 · ollama-js/0.5",
    )
    assert negotiator.clean_client_text("app\x00\nname " * 40).startswith("app")
    assert len(negotiator.clean_client_text("x" * 500)) == 128


def test_process_names_and_inference(negotiator, tmp):
    import subprocess
    script = os.path.join(tmp, "worker_app.py")
    with open(script, "w", encoding="utf-8") as stream:
        stream.write("import time\ntime.sleep(30)\n")
    child = subprocess.Popen([sys.executable, script, "--token", "secret"])
    try:
        # The kernel fills /proc/PID/cmdline once the exec completes.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with open(f"/proc/{child.pid}/cmdline", "rb") as stream:
                if stream.read():
                    break
            time.sleep(0.02)
        name = negotiator.process_name(child.pid)
        interpreter = os.path.basename(sys.executable)
        assert name == f"{interpreter} worker_app.py", name
        assert "secret" not in name
    finally:
        child.kill()
        child.wait()
    key = negotiator.client_key_and_label
    assert key({"unit": "vte-spawn-1.scope", "user": "roko",
                "candidate_processes": ["bash", "claude", "curl"]}) == (
        "unit:roko:vte-spawn-1.scope", "claude / curl (roko)",
    )
    assert key({"unit": "session-3.scope", "user": "roko",
                "candidate_processes": ["bash", "sshd"]}) == (
        "unit:roko:session-3.scope", "bash / sshd (roko)",
    )


def test_docker_directory_cache(negotiator):
    directory = negotiator.DockerDirectory("/nonexistent/docker.sock")
    # An unreachable Docker API degrades to "unknown", never an error.
    assert directory.lookup(ip="172.18.0.3") == ""
    directory.by_ip = {"172.18.0.3": "voryn"}
    directory.by_id = {"3b73baae0c53aabbcc": "moshi"}
    assert directory.lookup(ip="172.18.0.3") == "voryn"
    assert directory.lookup(container_id="3b73baae0c53") == "moshi"


def main():
    negotiator = load_negotiator(sys.argv[1])
    test_parse_socket_owner(negotiator)
    test_cgroup_unit(negotiator)
    test_keys_and_labels(negotiator)
    with tempfile.TemporaryDirectory() as tmp:
        test_process_names_and_inference(negotiator, tmp)
    test_docker_directory_cache(negotiator)
    print("client identity: PASS (socket owner, cgroup units, keys, labels, "
          "process names without arguments, "
          "docker directory)")


if __name__ == "__main__":
    main()
