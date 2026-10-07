#!/usr/bin/env python3
"""CPU-only regressions for exclusive, observed multi-GPU Ollama lanes."""
import concurrent.futures
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import threading
import unittest
from unittest import mock


HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/ollama-unify-multi-gpu'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/policy.json'


def load_module(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


n = load_module('multi_gpu_negotiator', HELPER)
p = load_module('multi_gpu_pool', pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
PAIR = ['GPU-large-0', 'GPU-large-1']
LARGE = 'fixture-sharded:latest'
LARGE_TAG = {'name': LARGE, 'model': LARGE, 'size': 130 * 1024**3,
             'capabilities': ['completion']}
USAGE = {PAIR[0]: 62000, PAIR[1]: 69000}


def harness(**kwargs):
    options = dict(max_servers=4, tags=[p.MODEL, p.OTHER_MODEL, LARGE_TAG],
                   runner_vram_by_gpu=USAGE)
    options.update(kwargs)
    return p.PoolHarness(HELPER, FIXTURE_BIN, **options)


def starts(case):
    return [event for event in p.events(case.event_log) if event['kind'] == 'start']


def acquire(case, scope):
    return p.control(case.socket_path, {
        'action': 'acquire', 'owner': 'multi-gpu-regression',
        'requested_mib': 1024, 'ttl': 30,
        'justification': 'CPU fixture verifies complete scope exclusion',
        'expected_duration_seconds': 30, 'gpu_uuids': scope,
    })['lease']['token']


class MultiGpuIntegrationTests(unittest.TestCase):
    def test_oversized_model_loads_one_exact_group_with_observed_accounting(self):
        with harness() as case:
            code, capacity, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 200, capacity)
            self.assertEqual(len(capacity['lanes']), 1)
            lane = capacity['lanes'][0]
            self.assertEqual(lane['gpu_uuids'], PAIR)
            self.assertEqual(lane['reserved_mib_by_gpu'], {gpu: 81920 for gpu in PAIR})
            self.assertEqual(lane['observed_vram_mib_by_gpu'], USAGE)
            self.assertEqual(starts(case)[0]['gpu'], ','.join(PAIR))
            self.assertEqual(starts(case)[0]['sched_spread'], '1')
            result = p.chat(case.proxy_port, LARGE, 'sharded-response', gpu_uuids=PAIR)
            self.assertEqual(result[0], 200, result)
            self.assertEqual(result[1]['gpu'], ','.join(PAIR))
            self.assertEqual(result[2]['X-Ollama-Unify-Lane'], lane['id'])
            self.assertEqual(len(starts(case)), 1)

    def test_group_order_is_requested_and_partial_scope_cannot_reuse_it(self):
        with harness() as case:
            requested = list(reversed(PAIR))
            code, capacity, _ = case.capacity(LARGE, gpu_uuids=requested)
            self.assertEqual(code, 200, capacity)
            self.assertEqual(capacity['lanes'][0]['gpu_uuids'], requested)
            code, payload, _ = case.capacity(LARGE, gpu_uuids=[PAIR[0]])
            self.assertEqual(code, 422, payload)
            self.assertFalse(payload['retryable'])
            self.assertEqual(len(starts(case)), 1)

    def test_preferences_cannot_widen_request_and_policy_needs_entire_group(self):
        with harness(model_gpu_preferences={LARGE: ['GPU-large-2', *reversed(PAIR)]}) as case:
            p.control(case.socket_path, {'action': 'set_model_gpus', 'model': LARGE,
                                         'gpu_uuids': [PAIR[0]]})
            code, payload, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 422, payload)
            self.assertFalse(starts(case))
            p.control(case.socket_path, {'action': 'set_model_gpus', 'model': LARGE,
                                         'gpu_uuids': [*PAIR, 'GPU-large-2']})
            code, capacity, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 200, capacity)
            self.assertEqual(set(capacity['lanes'][0]['gpu_uuids']), set(PAIR))
            self.assertNotIn('GPU-large-2', starts(case)[0]['gpu'].split(','))
            moved = p.control(case.socket_path, {'action': 'set_model_gpus', 'model': LARGE,
                                                 'gpu_uuids': [PAIR[0]]})
            self.assertIn(capacity['lanes'][0]['id'], moved['stopping_lanes'])
            p.wait_until(lambda: not p.managed_lanes(case.status()),
                         'policy removes whole incompatible group')

    def test_single_gpu_model_keeps_single_gpu_path(self):
        with harness() as case:
            code, capacity, _ = case.capacity(p.MODEL, gpu_uuids=PAIR)
            self.assertEqual(code, 200, capacity)
            self.assertEqual(len(capacity['lanes'][0]['gpu_uuids']), 1)
            self.assertNotIn(',', starts(case)[0]['gpu'])
            self.assertEqual(starts(case)[0]['sched_spread'], '0')

    def test_split_runner_removes_inherited_p2p_and_unified_memory_flags(self):
        with mock.patch.dict(os.environ, {'GGML_CUDA_P2P': '0',
                                          'GGML_CUDA_ENABLE_UNIFIED_MEMORY': '1'}), harness() as case:
            code, payload, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 200, payload)
            self.assertIsNone(starts(case)[0]['cuda_p2p'])
            self.assertIsNone(starts(case)[0]['unified_memory'])
            self.assertEqual(starts(case)[0]['no_pinned'], '1')

    def test_foreign_context_on_any_member_blocks_whole_group(self):
        for blocked in PAIR:
            with self.subTest(blocked=blocked), harness() as case:
                p.write_compute_apps(case.compute_apps, [(os.getpid(), blocked, 1)])
                code, payload, _ = case.capacity(LARGE, gpu_uuids=PAIR)
                self.assertEqual(code, 503, payload)
                self.assertTrue(payload['retryable'])
                self.assertFalse(starts(case))

    def test_pending_and_active_singleton_lease_exclude_sharded_member(self):
        with harness() as case:
            token = acquire(case, [PAIR[1]])
            for state in ('pending', 'active'):
                with self.subTest(state=state):
                    if state == 'active':
                        p.control(case.socket_path, {'action': 'ready', 'token': token})
                    code, payload, _ = case.capacity(LARGE, gpu_uuids=PAIR)
                    self.assertEqual(code, 503, payload)
                    self.assertTrue(payload['retryable'])
                    self.assertFalse(starts(case))
            p.control(case.socket_path, {'action': 'release', 'token': token})
            self.assertEqual(case.capacity(LARGE, gpu_uuids=PAIR)[0], 200)

    def test_group_drains_intersecting_singleton_before_warm_load(self):
        with harness(runner_vram_by_gpu={}) as case:
            code, first, _ = case.capacity(p.MODEL, gpu_uuids=[PAIR[1]])
            self.assertEqual(code, 200, first)
            code, capacity, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 200, capacity)
            events = p.events(case.event_log)
            stopped = next(i for i, event in enumerate(events)
                           if event['kind'] == 'stop' and event['gpu'] == PAIR[1])
            grouped = next(i for i, event in enumerate(events)
                           if event['kind'] == 'start' and event['gpu'] == ','.join(PAIR))
            self.assertLess(stopped, grouped)
            self.assertEqual([lane['id'] for lane in p.managed_lanes(case.status())],
                             [capacity['lanes'][0]['id']])

    def test_busy_group_cannot_be_reclaimed_or_have_member_sibling_started(self):
        with harness() as case:
            self.assertEqual(case.capacity(LARGE, gpu_uuids=PAIR)[0], 200)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                active = executor.submit(p.chat, case.proxy_port, LARGE,
                                         'group-busy', 0.8, 8, gpu_uuids=PAIR)
                p.wait_until(lambda: any(e.get('request_id') == 'group-busy'
                                        for e in p.request_events(case.event_log)),
                             'group request starts')
                for member in PAIR:
                    code, payload, _ = case.capacity(p.MODEL, gpu_uuids=[member])
                    self.assertEqual(code, 503, payload)
                code, outside, _ = case.capacity(p.MODEL, gpu_uuids=['GPU-large-2'])
                self.assertEqual(code, 200, outside)
                self.assertEqual([event['gpu'] for event in starts(case)],
                                 [','.join(PAIR), 'GPU-large-2'])
                self.assertFalse([event for event in p.events(case.event_log)
                                  if event['kind'] == 'stop'])
                self.assertEqual(active.result(timeout=8)[0], 200)

    def test_idle_group_reclaim_stops_all_members_before_singleton_start(self):
        for member in PAIR:
            with self.subTest(member=member), harness(max_servers=1) as case:
                code, capacity, _ = case.capacity(LARGE, gpu_uuids=PAIR)
                self.assertEqual(code, 200, capacity)
                code, replacement, _ = case.capacity(p.MODEL, gpu_uuids=[member])
                self.assertEqual(code, 200, replacement)
                observed = p.events(case.event_log)
                stopped = next(i for i, event in enumerate(observed)
                               if event['kind'] == 'stop' and event['gpu'] == ','.join(PAIR))
                started = next(i for i, event in enumerate(observed)
                               if event['kind'] == 'start' and event['gpu'] == member)
                self.assertLess(stopped, started)
                self.assertEqual([lane['id'] for lane in p.managed_lanes(case.status())],
                                 [replacement['lanes'][0]['id']])
                self.assertEqual(replacement['lanes'][0]['gpu_uuids'], [member])

    def test_foreign_peer_on_other_member_prevents_idle_group_reclaim(self):
        with harness(max_servers=1) as case:
            code, capacity, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 200, capacity)
            p.write_compute_apps(case.compute_apps, [(os.getpid(), PAIR[1], 1)])
            try:
                code, payload, _ = case.capacity(p.MODEL, gpu_uuids=[PAIR[0]])
                self.assertEqual(code, 503, payload)
                self.assertEqual([lane['id'] for lane in p.managed_lanes(case.status())],
                                 [capacity['lanes'][0]['id']])
                self.assertEqual(len(starts(case)), 1)
                self.assertFalse([event for event in p.events(case.event_log)
                                  if event['kind'] == 'stop'])
            finally:
                p.write_compute_apps(case.compute_apps, [])

    def test_idle_group_switch_stops_old_group_before_loading_next_model(self):
        other = dict(LARGE_TAG, name='fixture-sharded-other:latest', model='fixture-sharded-other:latest')
        with harness(max_servers=1, tags=[LARGE_TAG, other]) as case:
            self.assertEqual(case.capacity(LARGE, gpu_uuids=PAIR)[0], 200)
            code, replacement, _ = case.capacity(other['name'], gpu_uuids=PAIR)
            self.assertEqual(code, 200, replacement)
            lifecycle = [event['kind'] for event in p.events(case.event_log)
                         if event['kind'] in ('start', 'stop')]
            self.assertEqual(lifecycle, ['start', 'stop', 'start'])
            self.assertEqual([lane['model'] for lane in p.managed_lanes(case.status())],
                             [other['name']])

    def test_scoped_external_acquire_drains_entire_group(self):
        with harness() as case:
            code, capacity, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 200, capacity)
            token = acquire(case, [PAIR[1]])
            self.assertFalse(p.managed_lanes(case.status()))
            self.assertEqual([event['gpu'] for event in p.events(case.event_log)
                              if event['kind'] == 'stop'], [','.join(PAIR)])
            p.control(case.socket_path, {'action': 'release', 'token': token})

    def test_missing_or_outside_placement_is_rejected_before_ready(self):
        for observed in ([PAIR[0]], [*PAIR, 'GPU-large-2']):
            with self.subTest(observed=observed), harness(runner_gpu_scope=observed) as case:
                code, payload, _ = case.capacity(LARGE, gpu_uuids=PAIR)
                self.assertGreaterEqual(code, 400, payload)
                self.assertFalse(p.managed_lanes(case.status()))
                p.wait_until(lambda: bool([e for e in p.events(case.event_log)
                                          if e['kind'] == 'stop']), 'failed group exits')

    def test_warm_failure_does_not_publish_or_leak_group(self):
        rejected = dict(LARGE_TAG, name='fixture-reject:sharded', model='fixture-reject:sharded')
        with harness(tags=[rejected]) as case:
            code, payload, _ = case.capacity(rejected['name'], gpu_uuids=PAIR)
            self.assertGreaterEqual(code, 400, payload)
            self.assertFalse(p.managed_lanes(case.status()))
            self.assertEqual(len(starts(case)), 1)
            p.wait_until(lambda: bool([e for e in p.events(case.event_log)
                                      if e['kind'] == 'stop']), 'warm failure exits')

    def test_partial_cpu_offload_cannot_claim_ready_split_lane(self):
        with harness(partial_gpu_residency=True) as case:
            code, payload, _ = case.capacity(LARGE, gpu_uuids=PAIR)
            self.assertEqual(code, 503, payload)
            self.assertEqual(payload['reason_code'], 'gpu_runtime_unavailable')
            self.assertFalse(payload['retryable'])
            self.assertFalse(p.managed_lanes(case.status()))


class MultiGpuLifecycleTests(unittest.TestCase):
    def setUp(self):
        for name, value in [('BACKEND_TYPE', 'cuda'), ('SELECTED_GPUS', PAIR)]:
            patch = mock.patch.object(n, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.broker = n.Broker()
        self.lane = n.Lane('group', 'managed', '127.0.0.1', 1, PAIR[0],
                           LARGE, 1, 2 * 81920, 0, 0, mock.Mock(),
                           gpu_uuids=tuple(PAIR),
                           reserved_mib_by_gpu={gpu: 81920 for gpu in PAIR},
                           observed_vram_mib_by_gpu=USAGE.copy())
        self.broker.lanes[self.lane.lane_id] = self.lane
        alive = mock.patch.object(n, 'process_group_alive', return_value=True)
        alive.start()
        self.addCleanup(alive.stop)

    def test_failed_process_exit_retains_all_member_reservations(self):
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            n, 'unload_models_at', return_value=[]
        ), mock.patch.object(self.broker, '_terminate_process', return_value=False):
            failed = self.broker._stop_lanes([self.lane], 'fixture surviving descendant')
        self.assertEqual(failed, [self.lane])
        self.assertIs(self.broker.lanes['group'], self.lane)
        self.assertTrue(self.lane.retiring)
        self.assertEqual(self.lane.scope, tuple(PAIR))
        self.assertEqual(self.lane.reserved_mib_by_gpu, {gpu: 81920 for gpu in PAIR})

    def test_failed_warmup_and_surviving_process_keep_complete_loading_scope(self):
        self.broker.lanes.pop('group')
        process = mock.Mock(pid=12345)
        process.poll.return_value = None
        with mock.patch.object(n, 'require_gpu_health'), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ), mock.patch.object(n.os, 'access', return_value=True), mock.patch.object(
            n.subprocess, 'Popen', return_value=process
        ), mock.patch.object(n, 'backend_json_at', side_effect=[
            {'version': 'fixture'}, n.CapacityError('fixture warm-up failed')
        ]), mock.patch.object(n, 'unload_models_at'), mock.patch.object(
            self.broker, '_terminate_process', return_value=False
        ):
            with self.assertRaises(n.CapacityError):
                self.broker._spawn_lane(LARGE, tuple(PAIR), 134144,
                                        {'completion'}, '/api/generate',
                                        reserved_mib_by_gpu={gpu: 81920 for gpu in PAIR})
        retained = [lane for lane in self.broker.lanes.values() if lane.kind == 'managed']
        self.assertEqual(len(retained), 1)
        self.assertTrue(retained[0].retiring)
        self.assertFalse(retained[0].loading)
        self.assertEqual(retained[0].scope, tuple(PAIR))
        self.assertEqual(retained[0].reserved_mib_by_gpu, {gpu: 81920 for gpu in PAIR})

    def test_already_exited_lane_cannot_unload_a_reused_backend_port(self):
        with mock.patch.object(n, 'process_group_alive', return_value=False), mock.patch.object(
            n, 'unload_models_at'
        ) as unload, mock.patch.object(self.broker, '_terminate_process') as terminate:
            self.assertEqual(self.broker._stop_lanes([self.lane], 'already exited'), [])
        self.assertNotIn('group', self.broker.lanes)
        unload.assert_not_called()
        terminate.assert_not_called()

    def test_leader_exit_does_not_release_scope_until_all_descendants_exit(self):
        self.lane.process.poll.return_value = 0
        self.broker._prune_dead_lanes_locked()
        self.assertIs(self.broker.lanes['group'], self.lane)
        with mock.patch.object(n, 'process_group_alive', return_value=False):
            self.broker._prune_dead_lanes_locked()
        self.assertNotIn('group', self.broker.lanes)

    def test_foreign_member_defers_stop_cancel_and_native_unload(self):
        for member in PAIR:
            with self.subTest(member=member), mock.patch.object(
                n, 'foreign_gpu_usage', return_value={f'123@{member}': 1}
            ), mock.patch.object(n, 'unload_models_at') as unload, mock.patch.object(
                self.broker, '_terminate_process'
            ) as terminate:
                self.assertEqual(self.broker._stop_lanes([self.lane], 'foreign member'),
                                 [self.lane])
                self.broker._stop_expired_lane(self.lane)
                with self.assertRaises(n.CapacityError):
                    self.broker.prepare_managed_body(
                        self.lane, '/api/generate',
                        b'{"model":"fixture-sharded:latest","keep_alive":0}')
                unload.assert_not_called()
                terminate.assert_not_called()

    def test_misplaced_selected_gpu_stays_protected_while_foreign_peer_is_live(self):
        self.lane.observed_vram_mib_by_gpu['GPU-large-2'] = 1024
        with mock.patch.object(n, 'SELECTED_GPUS', [*PAIR, 'GPU-large-2']), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={'123@GPU-large-2': 1}
        ), mock.patch.object(n, 'unload_models_at') as unload, mock.patch.object(
            self.broker, '_terminate_process'
        ) as terminate:
            self.assertEqual(self.broker._peer_reserved_gpus_locked(),
                             {*PAIR, 'GPU-large-2'})
            self.assertEqual(self.broker._stop_lanes([self.lane], 'misplaced with peer'),
                             [self.lane])
        self.assertIs(self.broker.lanes['group'], self.lane)
        self.assertEqual(self.lane.reserved_mib_by_gpu, {gpu: 81920 for gpu in PAIR})
        unload.assert_not_called()
        terminate.assert_not_called()

    def test_unmonitored_placement_cannot_be_unloaded_or_killed(self):
        self.lane.observed_vram_mib_by_gpu['GPU-not-selected'] = 1024
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            n, 'unload_models_at'
        ) as unload, mock.patch.object(self.broker, '_terminate_process') as terminate:
            with self.assertRaises(n.CapacityError) as raised:
                self.broker._require_safe_gpu_transition(self.lane.protected_scope, 'group')
            self.assertEqual(raised.exception.reason_code, 'gpu_placement_unverified')
            self.assertFalse(raised.exception.retryable)
            self.assertEqual(self.broker._stop_lanes([self.lane], 'unmonitored placement'),
                             [self.lane])
        self.assertIs(self.broker.lanes['group'], self.lane)
        unload.assert_not_called()
        terminate.assert_not_called()

    def test_sibling_group_transitions_block_on_every_member(self):
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}):
            for member in PAIR:
                with self.subTest(member=member), self.assertRaises(n.CapacityError):
                    self.broker._require_safe_gpu_transition(member)
            self.broker._require_safe_gpu_transition(tuple(PAIR), ignore_lane_id='group')

    def test_loading_group_owns_all_members_but_cannot_route_inference(self):
        self.lane.loading = True
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}):
            self.assertEqual(self.broker._peer_reserved_gpus_locked(), set(PAIR))
            self.assertIsNone(self.broker._select_lane_locked(LARGE, True, PAIR))
            self.assertEqual(self.lane.public_summary()['state'], 'loading')

    def test_legacy_unscoped_external_lease_blocks_every_group_member(self):
        now = n.time.time()
        for state in ('pending', 'active', 'revoking'):
            with self.subTest(state=state):
                self.broker.leases['legacy'] = n.Lease(
                    'legacy', 'external', state, 1024, now, now, now, 60, {}, [])
                with mock.patch.object(n, 'foreign_gpu_usage', return_value={}):
                    with self.assertRaises(n.CapacityError):
                        self.broker._ensure_group_capacity(
                            LARGE, 1, 134144, {'completion'}, '/api/generate',
                            tuple(PAIR), None)

    def test_live_legacy_scope_conversion_rejects_group_overlap_before_mutation(self):
        now = n.time.time()
        lease = n.Lease('legacy', 'external', 'active', 1024,
                        now, now, now, 60, {}, [])
        self.broker.leases['legacy'] = lease
        devices = [{'uuid': gpu, 'total_mib': 81920, 'free_mib': 80000}
                   for gpu in PAIR]
        with mock.patch.object(n, 'require_gpu_health'), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ), mock.patch.object(n, 'gpu_snapshot', return_value=devices), mock.patch.object(
            self.broker, '_persist_leases_locked'
        ) as persist, mock.patch.object(self.broker, '_terminate_process') as terminate:
            with self.assertRaises(RuntimeError):
                self.broker.scope('legacy', [PAIR[1]])
        self.assertEqual(lease.gpu_uuids, [])
        persist.assert_not_called()
        terminate.assert_not_called()

    def test_operator_teardown_finishes_before_external_scope_is_granted(self):
        self.lane.gpu_uuids = (PAIR[0],)
        self.lane.reserved_mib_by_gpu = {PAIR[0]: 4096}
        self.lane.observed_vram_mib_by_gpu = {PAIR[0]: 2048}
        entered = threading.Event()
        release_stop = threading.Event()
        acquire_started = threading.Event()
        granted = threading.Event()
        effects_after_grant = []
        original = self.broker._require_safe_gpu_transition

        def pause_after_check(scope, ignore_lane_id=None):
            original(scope, ignore_lane_id)
            if ignore_lane_id == 'group':
                entered.set()
                if not release_stop.wait(3):
                    raise RuntimeError('fixture stop barrier expired')

        def external_acquire():
            acquire_started.set()
            result = self.broker.acquire('external-fixture', 1024, 60, [PAIR[0]],
                                         'CPU fixture transition ordering', 60)
            granted.set()
            return result

        def observe_unload(*_args, **_kwargs):
            if granted.is_set():
                effects_after_grant.append('unload')
            return []

        devices = [{'uuid': gpu, 'total_mib': 81920, 'free_mib': 80000}
                   for gpu in PAIR]
        with mock.patch.object(n, 'require_gpu_health'), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ), mock.patch.object(n, 'gpu_snapshot', return_value=devices), mock.patch.object(
            self.broker, '_unload_base_models', return_value=[]
        ), mock.patch.object(self.broker, 'begin_drain'), mock.patch.object(
            self.broker, '_persist_leases_locked'
        ), mock.patch.object(self.broker, '_require_safe_gpu_transition', side_effect=pause_after_check), mock.patch.object(
            n, 'unload_models_at', side_effect=observe_unload
        ), mock.patch.object(self.broker, '_terminate_process', return_value=True), concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            stopped = executor.submit(self.broker.stop_lane, 'group')
            try:
                self.assertTrue(entered.wait(2))
                acquired = executor.submit(external_acquire)
                self.assertTrue(acquire_started.wait(2))
                self.assertFalse(granted.wait(0.15), 'lease granted before teardown completed')
            finally:
                release_stop.set()
            self.assertEqual(stopped.result(timeout=3)['stopped_lanes'], ['group'])
            self.assertEqual(acquired.result(timeout=3)['lease']['state'], 'pending')
        self.assertFalse(effects_after_grant)

    def test_pending_external_scope_defers_operator_stop(self):
        self.lane.gpu_uuids = (PAIR[0],)
        self.lane.reserved_mib_by_gpu = {PAIR[0]: 4096}
        self.lane.observed_vram_mib_by_gpu = {PAIR[0]: 2048}
        now = n.time.time()
        self.broker.leases['pending'] = n.Lease('pending', 'external', 'pending',
                                                1024, now, now, now, 60, {}, [PAIR[0]])
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            n, 'unload_models_at'
        ) as unload, mock.patch.object(self.broker, '_terminate_process') as terminate:
            with self.assertRaises(RuntimeError):
                self.broker.stop_lane('group')
        self.assertIs(self.broker.lanes['group'], self.lane)
        unload.assert_not_called()
        terminate.assert_not_called()

    def test_cancellation_defers_without_blocking_a_lease_transition(self):
        now = n.time.monotonic()
        active = n.ActiveRequest('cancelled', self.lane, '', now, now, now + 1,
                                 cancel_requested_at=now, lane_stop_started=True)
        self.lane.in_flight = 1
        self.broker.active_request_records['cancelled'] = active
        self.broker.active_requests = 1
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            self.broker, '_terminate_process', return_value=False
        ) as terminate, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            with self.broker.transition:
                stopped = executor.submit(self.broker._stop_expired_lane, self.lane)
                stopped.result(timeout=0.5)
                terminate.assert_not_called()
        self.assertIs(self.broker.active_request_records['cancelled'], active)
        self.assertEqual(self.broker.active_requests, 1)
        self.assertEqual(self.lane.in_flight, 1)
        self.assertFalse(active.lane_stop_started)
        self.assertGreater(active.expires_at, now)

    def test_active_unscoped_external_owner_blocks_singleton_transitions(self):
        self.lane.gpu_uuids = (PAIR[0],)
        self.lane.reserved_mib_by_gpu = {PAIR[0]: 4096}
        self.lane.observed_vram_mib_by_gpu = {PAIR[0]: 2048}
        now = n.time.time()
        self.broker.leases['legacy'] = n.Lease('legacy', 'external', 'active',
                                              1024, now, now, now, 60, {}, [])
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            n, 'unload_models_at'
        ) as unload, mock.patch.object(self.broker, '_terminate_process') as terminate, mock.patch.object(
            n, 'running_models', return_value=[{'name': LARGE}]
        ):
            self.assertIsNone(self.broker._select_lane_locked(LARGE, True, [PAIR[0]]))
            self.assertEqual(self.broker._stop_lanes([self.lane], 'unscoped live owner'), [self.lane])
            with self.assertRaises(n.CapacityError):
                self.broker._unload_base_models()
            unload.assert_not_called()
            terminate.assert_not_called()

    def test_unscoped_external_ownership_cannot_overlap_another_lease(self):
        now = n.time.time()
        for existing_scope, requested_scope in (([], [PAIR[0]]), ([PAIR[0]], None)):
            with self.subTest(existing_scope=existing_scope, requested_scope=requested_scope):
                existing = n.Lease('existing', 'external', 'active', 1024,
                                   now, now, now, 60, {}, existing_scope)
                self.broker.leases = {'existing': existing}
                with mock.patch.object(n, 'require_gpu_health'), mock.patch.object(
                    self.broker, 'begin_drain'
                ) as drain, mock.patch.object(self.broker, '_unload_base_models') as unload, mock.patch.object(
                    self.broker, '_persist_leases_locked'
                ) as persist:
                    with self.assertRaises(RuntimeError):
                        self.broker.acquire('new-owner', 1024, 60, requested_scope,
                                            'CPU fixture verifies external exclusivity', 60)
                self.assertEqual(self.broker.leases, {'existing': existing})
                drain.assert_not_called()
                unload.assert_not_called()
                persist.assert_not_called()

    def test_legacy_release_never_unloads_or_kills_while_external_cuda_may_remain(self):
        now = n.time.time()
        for force in (False, True):
            with self.subTest(force=force):
                self.broker.leases['legacy'] = n.Lease('legacy', 'external', 'active',
                                                      1024, now, now, now, 60, {}, [])
                with mock.patch.object(self.broker, 'begin_drain'), mock.patch.object(
                    n, 'wait_for_foreign_settle'
                ) as settle, mock.patch.object(self.broker, '_persist_leases_locked'), mock.patch.object(
                    n, 'gpu_snapshot', return_value=[]
                ), mock.patch.object(n, 'host_memory_snapshot', return_value={}), mock.patch.object(
                    self.broker, 'stop_pool_lanes'
                ) as stop, mock.patch.object(self.broker, '_unload_base_models') as unload, mock.patch.object(
                    self.broker, '_terminate_process'
                ) as terminate:
                    result = self.broker.release('legacy', force=force)
                self.assertEqual(result['stopped_lanes'], [])
                self.assertEqual(result['unloaded'], [])
                self.assertNotIn('legacy', self.broker.leases)
                self.assertIs(self.broker.lanes['group'], self.lane)
                self.assertEqual(settle.call_count, 0 if force else 1)
                stop.assert_not_called()
                unload.assert_not_called()
                terminate.assert_not_called()

    def test_full_scope_allowlist_and_reservations_are_not_scalar(self):
        self.assertTrue(self.lane.allows(PAIR))
        self.assertFalse(self.lane.allows([PAIR[0]]))
        devices = [{'uuid': gpu, 'total_mib': 81920, 'free_mib': 80000,
                    'used_mib': 1920} for gpu in PAIR]
        with mock.patch.object(n, 'gpu_snapshot', return_value=devices), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ):
            self.assertEqual(self.broker._placement_devices(set()), [])
            with mock.patch.object(self.broker, '_peer_reserved_gpus_locked', return_value=set()):
                placement = self.broker._placement_devices(set())
        self.assertEqual({item['uuid']: item['free_mib'] for item in placement},
                         {gpu: 0 for gpu in PAIR})


if __name__ == '__main__':
    unittest.main()
