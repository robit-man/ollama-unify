#!/usr/bin/env python3
"""Regressions for free VRAM on a node whose driver requires recovery."""
import importlib.machinery
import importlib.util
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/ollama-unify-negotiator'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/policy.json'
loader = importlib.machinery.SourceFileLoader('health_negotiator', sys.argv.pop(1))
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)


class GpuHealthTests(unittest.TestCase):
    def setUp(self):
        self.backend = mock.patch.object(n, 'BACKEND_TYPE', 'cuda')
        self.selected = mock.patch.object(n, 'SELECTED_GPUS', ['GPU-selected'])
        self.backend.start()
        self.selected.start()
        self.cache = mock.patch.object(n, '_GPU_HEALTH_CACHE', None)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.addCleanup(self.backend.stop)
        self.addCleanup(self.selected.stop)

    @contextlib.contextmanager
    def query(self, stdout, returncode=0, stderr=''):
        with mock.patch.object(n, '_GPU_HEALTH_CACHE', None), mock.patch.object(
            n.subprocess, 'run', return_value=subprocess.CompletedProcess([], returncode, stdout, stderr)
        ) as run:
            yield run

    def test_polling_is_coalesced_but_growth_forces_fresh_telemetry(self):
        with self.query('GPU-selected, None\n') as run, mock.patch.object(n.time, 'monotonic', return_value=100) as clock:
            n.require_gpu_health()
            n.require_gpu_health()
            self.assertEqual(run.call_count, 1)
            n.require_gpu_health(refresh=True)
            self.assertEqual(run.call_count, 2)
            clock.return_value = 100.6
            run.return_value.stdout = 'GPU-selected, Reboot\n'
            with self.assertRaises(n.PermanentCapacityError):
                n.require_gpu_health()
            self.assertEqual(run.call_count, 3)

    def test_none_and_unsupported_device_are_admissible(self):
        for action in ('None', 'N/A', '[N/A]'):
            with self.subTest(action=action), self.query('GPU-selected, ' + action + '\n'):
                self.assertFalse(n.gpu_health_snapshot()['admission_blocked'])
                n.require_gpu_health()

    def test_recovery_actions_block_and_preserve_request_identity(self):
        for action in ('Reset', 'Reboot', 'Drain P2P', 'Drain and Reset', '[Unknown Error]'):
            with self.subTest(action=action), self.query('GPU-selected, ' + action + '\n'):
                with self.assertRaises(n.PermanentCapacityError) as raised:
                    n.require_gpu_health(request_id='request-1', logical_request_id='turn-1')
                exc = raised.exception
                self.assertEqual((exc.status, exc.reason_code, exc.retryable),
                                 (503, 'gpu_recovery_required', False))
                self.assertEqual((exc.request_id, exc.logical_request_id), ('request-1', 'turn-1'))

    def test_reboot_on_unselected_gpu_blocks_selected_gpu(self):
        with self.query('GPU-selected, None\nGPU-display, Reboot\n'):
            self.assertEqual(n.gpu_health_snapshot()['recovery_actions'], {'GPU-display': 'Reboot'})
            with self.assertRaises(n.PermanentCapacityError):
                n.require_gpu_health()
        with self.query('GPU-selected, None\nGPU-display, Reset\n'):
            n.require_gpu_health()

    def test_failed_or_partial_telemetry_is_not_free_capacity(self):
        for stdout, code in [('', 0), ('GPU-other, None\n', 0), ('GPU-selected, None\n', 255)]:
            with self.subTest(stdout=stdout, code=code), self.query(stdout, code):
                with self.assertRaises(n.CapacityError) as raised:
                    n.require_gpu_health()
                self.assertEqual(raised.exception.reason_code, 'gpu_health_unavailable')
        with mock.patch.object(n.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
            self.assertTrue(n.gpu_health_snapshot()['admission_blocked'])

    def test_partial_inventory_still_reports_confirmed_reboot(self):
        with self.query('Unable to determine handle for GPU2\nGPU-selected, Reboot\n', 255):
            with self.assertRaises(n.PermanentCapacityError) as raised:
                n.require_gpu_health()
            self.assertEqual(raised.exception.reason_code, 'gpu_recovery_required')

    def test_older_driver_is_explicitly_unsupported(self):
        with self.query('Field "gpu_recovery_action" is not a valid field to query.\n', 2):
            health = n.gpu_health_snapshot()
            self.assertFalse(health['supported'])
            self.assertFalse(health['admission_blocked'])
            self.assertTrue(n.gpu_health_warnings(health))
        # A generic query failure must not activate the compatibility fallback.
        with self.query('', 2, 'Failed to initialize NVML'):
            self.assertTrue(n.gpu_health_snapshot()['admission_blocked'])

    def test_other_backends_do_not_call_nvidia(self):
        with mock.patch.object(n, 'BACKEND_TYPE', 'rocm'), mock.patch.object(n.subprocess, 'run') as run:
            n.require_gpu_health()
            run.assert_not_called()

    def test_fault_blocks_lease_before_unload_or_reservation(self):
        broker = n.Broker()
        with self.query('GPU-selected, Reboot\n'), mock.patch.object(n, 'unload_all_models') as unload:
            with self.assertRaises(n.PermanentCapacityError):
                broker.acquire('health-regression', 1024, 60, ['GPU-selected'],
                               'Verify recovery refuses new allocations', 60)
            unload.assert_not_called()
        self.assertFalse(broker.leases)
        self.assertFalse(broker.draining)

    def test_fault_blocks_ready_growth_and_lane_start(self):
        broker = n.Broker()
        with self.query('GPU-selected, Reboot\n'), mock.patch.object(n.subprocess, 'Popen') as start:
            for operation in (
                lambda: broker.ready('pending-lease'),
                lambda: broker.prepare('active-lease'),
                lambda: broker.ensure_capacity('fixture:latest', 1),
                lambda: broker._spawn_lane('fixture:latest', 'GPU-selected', 1024, set(), ''),
            ):
                with self.assertRaises(n.PermanentCapacityError):
                    operation()
            start.assert_not_called()

    def test_fault_blocks_inference_but_allows_unload(self):
        broker = n.Broker()
        with mock.patch.object(n, 'POOL_ENABLED', False), self.query('GPU-selected, Reboot\n'):
            with self.assertRaises(n.PermanentCapacityError):
                broker.proxy_enter('fixture:latest', True)
            self.assertEqual(broker.active_requests, 0)
            admission = broker.proxy_enter('fixture:latest', True, allow_during_drain=True)
            broker.proxy_exit(admission, 'fixture:latest', True)
            self.assertEqual(broker.active_requests, 0)

    def test_fault_while_queued_releases_waiter(self):
        broker = n.Broker()
        healthy = {'admission_blocked': False}
        faulty = {'admission_blocked': True, 'recovery_actions': {'GPU-selected': 'Reboot'}}
        with mock.patch.object(n, 'POOL_ENABLED', True), mock.patch.object(n, 'gpu_health_snapshot', side_effect=[healthy, faulty]):
            with self.assertRaises(n.PermanentCapacityError):
                broker.proxy_enter('fixture:latest', True, request_id='queued-1', logical_request_id='turn-1')
        self.assertFalse(broker.waiters)
        self.assertFalse(broker.logical_in_flight)
        self.assertEqual(broker.retained_request_bytes, 0)

    def test_fault_does_not_block_owner_heartbeat_or_release(self):
        broker = n.Broker()
        now = time.time()
        broker.leases['existing'] = n.Lease('existing', 'owner', 'active', 1024,
                                          now, now, now, 60, {}, ['GPU-selected'])
        with mock.patch.object(n, 'gpu_health_snapshot', side_effect=AssertionError('cleanup must stay available')), mock.patch.object(
            broker, '_persist_leases_locked'
        ), mock.patch.object(n, 'wait_for_foreign_settle'), mock.patch.object(n, 'gpu_snapshot', return_value=[]), mock.patch.object(n, 'host_memory_snapshot', return_value={}):
            self.assertTrue(broker.heartbeat('existing')['ok'])
            self.assertTrue(broker.release('existing')['ok'])
        self.assertFalse(broker.leases)

    def test_cli_discovery_preserves_health_warnings_with_and_without_broker(self):
        health = {'supported': True, 'admission_blocked': True,
                  'recovery_actions': {'GPU-selected': 'Reboot'},
                  'missing_selected_gpu_ids': [], 'error': None}
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable), mock.patch.object(sys, 'argv', ['broker', 'discover']), mock.patch.object(
                n, 'discovery_document', return_value={'gpu_health': health, 'warnings': ['lease warning', *n.gpu_health_warnings(health)]}
            ), mock.patch.object(n, 'send_control', return_value={'leases': []}, side_effect=OSError('offline') if unavailable else None), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(n.main(), 0)
                document = json.loads(output.getvalue())
                self.assertEqual(len(document['warnings']), 2)
                self.assertIn('New GPU work is blocked', document['warnings'][1])


if __name__ == '__main__':
    unittest.main()
