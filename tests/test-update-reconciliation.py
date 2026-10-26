#!/usr/bin/env python3
"""Updates keep the backend topology without an obsolete release gate."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

REPO = Path(sys.argv.pop(1)).resolve()


class UpdateReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = self.root / 'calls'
        self.guard_calls = self.root / 'guard-calls'
        self.helper = self.root / 'version-guard'
        self.helper.write_text('#!/bin/sh\nprintf called >> "'+str(self.guard_calls)+'"\nexit 7\n')
        self.helper.chmod(0o755)
        self.script = self.root / 'ollama-unify.sh'
        self.script.write_text((REPO/'ollama-unify.sh').read_text().replace(
            '/usr/local/sbin/ollama-version-guard', str(self.helper)))
        self.env = os.environ.copy()
        self.env.update(OLLAMA_UNIFY_ASSUME_YES='1', TEST_CALLS=str(self.calls))
        self.env['PATH'] = str(REPO/'tests/fixtures/bin')+':'+self.env['PATH']

    def run_shell(self, body):
        return subprocess.run(['bash', '-c', body, 'test', str(self.script)],
                              capture_output=True, text=True, env=self.env, timeout=10)

    def mocked_command(self, command):
        return self.run_shell('''
source "$1"
record() { printf '%s\\n' "$*" >> "$TEST_CALLS"; }
banner() { :; }
build_safety_profile() { SAFETY_OLLAMA_BACKEND=127.0.0.1:11436; }
report_update_status() { :; }
report_policy_drift() { return 1; }
ollama_installed_version() { printf 0.34.4; }
ollama_latest_version() { printf 0.40.11; }
state_value() { printf 0.34.4; }
systemctl() { record "systemctl $*"; }
sudo() { if [ "$1" = -n ]; then return 0; fi; "$@"; }
curl() { printf 'printf "installer\\\\n" >> "$TEST_CALLS"\\n'; }
install_systemd_safety_policy() { record policy; }
cycle_negotiated_stack() { record cycle; }
verify_negotiated_stack() { record verify; }
HOST_SERVICE_MANAGER=systemd
HAS_CURL=1
HAS_SUDO=1
''' + command)

    def test_update_runs_with_an_obsolete_guard_present(self):
        result = self.mocked_command('update_ollama')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(self.calls.read_text().splitlines(), [
            'systemctl stop ollama-unify-negotiator.service',
            'systemctl stop ollama.service', 'installer', 'policy', 'cycle', 'verify'])
        self.assertFalse(self.guard_calls.exists())

    def test_manual_reconcile_keeps_its_service_order(self):
        result = self.mocked_command('reconcile_after_update')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls[-5:], ['systemctl stop ollama-unify-negotiator.service',
                                    'systemctl stop ollama.service', 'policy', 'cycle', 'verify'])
        self.assertFalse(self.guard_calls.exists())

    def render_reconciler(self, state):
        result = self.run_shell('source "$1"; render_reconcile_helper')
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.replace('/usr/local/share/ollama-unify/state.env', str(state))

    def test_reconciler_with_no_state_leaves_services_untouched(self):
        result = subprocess.run(['bash'], input=self.render_reconciler(self.root/'absent'),
                                capture_output=True, text=True, env=self.env, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nothing to reconcile', result.stdout)
        self.assertFalse(self.guard_calls.exists())

    def test_watchdog_repins_a_new_release_without_consulting_guard(self):
        state = self.root/'state.env'
        state.write_text('OLLAMA_UNIFY_PINNED_BACKEND="127.0.0.1:11436"\n'
                         'OLLAMA_UNIFY_OLLAMA_VERSION="0.34.4"\n')
        dropin = self.root/'dropin.conf'
        dropin.write_text('[Service]\nEnvironment="OLLAMA_HOST=127.0.0.1:11436"\n')
        config = self.root/'negotiator.conf'
        config.write_text('OLLAMA_UNIFY_BACKEND="127.0.0.1:11436"\n')
        script = self.render_reconciler(state).replace(
            '/etc/systemd/system/ollama.service.d/zzz-ollama-unify-safety.conf', str(dropin)
        ).replace('/etc/default/ollama-unify-negotiator', str(config))
        prefix = '''
systemctl() { printf '%s\\n' "$*" >> "$TEST_CALLS"; }
ollama() { printf 'ollama version is 0.40.11\\n'; }
'''
        result = subprocess.run(['bash'], input=prefix+script, capture_output=True,
                                text=True, env=self.env, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('OLLAMA_UNIFY_OLLAMA_VERSION="0.40.11"', state.read_text())
        self.assertEqual(self.calls.read_text().splitlines(), [
            'daemon-reload', 'stop ollama-unify-negotiator.service',
            'restart ollama.service', 'list-unit-files ollama-unify-negotiator.service',
            'start ollama-unify-negotiator.service'])
        self.assertFalse(self.guard_calls.exists())


if __name__ == '__main__':
    unittest.main()
