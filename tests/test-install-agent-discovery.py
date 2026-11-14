#!/usr/bin/env python3
"""Optional agent discovery must not interrupt safety-policy installation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(sys.argv.pop(1)).resolve()


class AgentDiscoveryInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent_home = self.root / "agent-home"
        self.agent_dir = self.agent_home / ".codex"
        self.calls = self.root / "calls"
        self.env = {**os.environ, "TEST_AGENT_HOME": str(self.agent_home),
                    "TEST_CALLS": str(self.calls), "TMPDIR": str(self.root)}

    def run_install(self, setup="", enabled="1"):
        body = r'''
source "$1/ollama-unify.sh"
SUDO_USER=fixture-agent
getent() {
  [ "${TEST_LOOKUP_FAIL:-0}" = 0 ] || return 2
  printf 'fixture-agent:x:1000:1000:Fixture:%s:/bin/bash\n' "$TEST_AGENT_HOME"
}
id() { printf 'fixture-group\n'; }
install() {
  [ "${TEST_WRITE_FAIL:-0}" = 0 ] || return 7
  command cp "${@: -2}"
}
''' + setup + r'''
install_global_codex_gpu_instructions ""
printf 'setup continued\n' >> "$TEST_CALLS"
'''
        env = {**self.env, "OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY": enabled}
        return subprocess.run(["bash", "-c", body, "test", str(REPO)],
                              env=env, capture_output=True, text=True, timeout=10)

    def assert_continued(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls.read_text(), "setup continued\n")

    def test_missing_agent_directory_keeps_setup_running(self):
        self.assert_continued(self.run_install())
        self.assertFalse(self.agent_dir.exists())

    def test_missing_agent_user_keeps_setup_running(self):
        self.assert_continued(self.run_install("unset SUDO_USER USER\n"))

    def test_missing_account_keeps_setup_running(self):
        self.assert_continued(self.run_install("TEST_LOOKUP_FAIL=1\n"))

    def test_missing_home_keeps_setup_running(self):
        self.assert_continued(self.run_install("TEST_AGENT_HOME=''\n"))

    def test_disabled_discovery_keeps_setup_running(self):
        self.assert_continued(self.run_install(enabled="0"))

    def test_symlink_skip_preserves_target_and_keeps_setup_running(self):
        self.agent_dir.mkdir(parents=True)
        target = self.root / "external-instructions"
        target.write_text("existing instructions\n")
        (self.agent_dir / "AGENTS.md").symlink_to(target)
        self.assert_continued(self.run_install())
        self.assertEqual(target.read_text(), "existing instructions\n")

    def test_existing_instructions_are_preserved_and_update_is_idempotent(self):
        self.agent_dir.mkdir(parents=True)
        instructions = self.agent_dir / "AGENTS.md"
        instructions.write_text("Local instructions\n")
        first = self.run_install()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        expected = instructions.read_text()
        second = self.run_install()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        actual = instructions.read_text()
        marker = "<!-- BEGIN ollama-unify GPU negotiator -->"
        self.assertEqual(actual.split(marker)[0].rstrip("\n"), "Local instructions")
        self.assertEqual(actual[actual.index(marker):], expected[expected.index(marker):])
        self.assertEqual(actual.count(marker), 1)

    def test_write_failure_still_stops_setup(self):
        self.agent_dir.mkdir(parents=True)
        result = self.run_install("TEST_WRITE_FAIL=1\n")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertFalse(self.calls.exists())

    def test_invalid_setting_still_stops_setup(self):
        result = self.run_install(enabled="invalid")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
