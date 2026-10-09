#!/usr/bin/env python3
"""CPU-only public HTTP controls for explicit exact-group placement of fitting models."""
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import sys
import unittest

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
loader = importlib.machinery.SourceFileLoader(
    'forced_group_pool', str(pathlib.Path(__file__).with_name('test-negotiator-pool.py')))
spec = importlib.util.spec_from_loader(loader.name, loader)
p = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = p
loader.exec_module(p)

GPUS = ['GPU-large-0', 'GPU-large-1', 'GPU-large-2']
PAIR = GPUS[:2]
CAPACITY = '/.well-known/ollama-unify-gpu-negotiator/capacity'


def harness(**overrides):
    options = dict(max_servers=3, auto_model_context=True,
                   runner_vram_by_gpu={gpu: 1024 for gpu in GPUS})
    options.update(overrides)
    return p.PoolHarness(HELPER, FIXTURE_BIN, **options)


def capacity(case, **overrides):
    body = {'model': p.MODEL, 'parallel': 1, 'endpoint': '/api/chat',
            'gpu_uuids': PAIR, 'placement': 'exclusive_group', **overrides}
    return p.http_json(case.proxy_port, 'POST', CAPACITY, body)


def starts(case):
    return [event for event in p.events(case.event_log) if event['kind'] == 'start']


def send(case, certificate, logical='forced-group:request', **overrides):
    return p.http_json(case.proxy_port, 'POST', '/api/chat', {
        'model': p.MODEL, 'stream': False, 'mock_request_id': logical,
        'options': {'num_ctx': 262144}, **overrides,
    }, extra_headers={
        'X-Ollama-Unify-GPU-UUIDs': ','.join(certificate['gpu_uuids']),
        'X-Ollama-Unify-Warm-Admission': json.dumps(certificate),
        'X-Ollama-Unify-Logical-Request-Id': logical,
    })


class ForcedGroupHTTPTests(unittest.TestCase):
    def test_fitting_model_uses_exact_ordered_group_without_profile_inflation(self):
        with harness() as case:
            ordered = list(reversed(PAIR))
            code, result, _ = capacity(case, gpu_uuids=ordered)
            self.assertEqual(code, 200, result)
            self.assertEqual(result['requested_placement'], 'exclusive_group')
            lane = result['lanes'][0]
            self.assertEqual(lane['gpu_uuids'], ordered)
            self.assertEqual(lane['reserved_mib_by_gpu'], {gpu: 81920 for gpu in PAIR})
            self.assertEqual(lane['observed_vram_mib_by_gpu'], {gpu: 1024 for gpu in PAIR})
            self.assertLess(lane['reserved_mib'], 81920)
            self.assertEqual(lane['resolved_context_length'], 262144)
            self.assertEqual(starts(case)[0]['gpu'], ','.join(ordered))
            self.assertEqual(starts(case)[0]['sched_spread'], '1')
            self.assertEqual(case.backend.tags[0]['size'], 1024**3)
            code, proof, _ = capacity(case, gpu_uuids=ordered, warm_admission_proof=True)
            self.assertEqual(code, 200, proof)
            self.assertEqual(proof['warm_admission_lanes'][0]['native_ps']['digest'], p.MODEL_DIGEST)
            response = send(case, proof['warm_admission'])
            self.assertEqual(response[0], 200, response)
            self.assertEqual(response[1]['gpu'], ','.join(ordered))
            self.assertEqual(len(starts(case)), 1)

    def test_explicit_three_member_group_cannot_shrink_to_two(self):
        with harness() as case:
            code, result, _ = capacity(case, gpu_uuids=[GPUS[2], GPUS[0], GPUS[1]])
            self.assertEqual(code, 200, result)
            self.assertEqual(result['lanes'][0]['gpu_uuids'], [GPUS[2], GPUS[0], GPUS[1]])
            self.assertEqual(set(result['lanes'][0]['reserved_mib_by_gpu']), set(GPUS))

    def test_absent_placement_still_prefers_singleton_and_group_proof_never_loads(self):
        with harness() as case:
            code, first, _ = case.capacity(p.MODEL, gpu_uuids=PAIR)
            self.assertEqual(code, 200, first)
            self.assertEqual(len(first['lanes'][0]['gpu_uuids']), 1)
            self.assertEqual(capacity(case, placement='auto')[1]['lanes'][0]['id'],
                             first['lanes'][0]['id'])
            code, failed, _ = capacity(case, warm_admission_proof=True)
            self.assertEqual(code, 503, failed)
            self.assertEqual(failed['reason_code'], 'warm_preflight_required')
            self.assertEqual(len(starts(case)), 1)
            code, grouped, _ = capacity(case)
            self.assertEqual(code, 200, grouped)
            self.assertNotEqual(grouped['lanes'][0]['id'], first['lanes'][0]['id'])
            self.assertEqual(grouped['lanes'][0]['gpu_uuids'], PAIR)

    def test_group_reuse_requires_exact_order_and_complete_membership(self):
        with harness() as case:
            code, first, _ = capacity(case)
            self.assertEqual(code, 200, first)
            self.assertEqual(capacity(case)[1]['lanes'][0]['id'], first['lanes'][0]['id'])
            for scope in (list(reversed(PAIR)), [GPUS[0], GPUS[2]], GPUS):
                with self.subTest(scope=scope):
                    code, proof, _ = capacity(case, gpu_uuids=scope, warm_admission_proof=True)
                    self.assertEqual(code, 503, proof)
            self.assertEqual(capacity(case, gpu_uuids=list(reversed(PAIR)))[0], 503)
            self.assertEqual(len(starts(case)), 1)

    def test_invalid_or_pruned_group_and_unsupported_placement_never_load(self):
        with harness() as case:
            for changes in (
                {'placement': 'spread'}, {'placement': None}, {'placement': 1},
                {'gpu_uuids': None}, {'gpu_uuids': PAIR[0]}, {'gpu_uuids': [PAIR[0]]},
                {'gpu_uuids': [PAIR[0], PAIR[0]]},
                {'gpu_uuids': [*PAIR, 'GPU-unselected']}, {'parallel': 2},
            ):
                with self.subTest(changes=changes):
                    code, result, _ = capacity(case, **changes)
                    self.assertIn(code, (400, 422), result)
            self.assertFalse(starts(case))

    def test_hard_policy_cannot_narrow_explicit_group_to_subset(self):
        with harness() as case:
            p.control(case.socket_path, {'action': 'set_model_gpus', 'model': p.MODEL,
                                        'gpu_uuids': [PAIR[0]]})
            code, result, _ = capacity(case)
            self.assertEqual(code, 409, result)
            self.assertEqual(result['reason_code'], 'gpu_policy_conflict')
            self.assertFalse(starts(case))

    def test_foreign_member_never_substitutes_other_selected_gpu(self):
        for member in PAIR:
            with self.subTest(member=member), harness() as case:
                p.write_compute_apps(case.compute_apps, [(os.getpid(), member, 1)])
                code, result, _ = capacity(case)
                self.assertEqual(code, 503, result)
                self.assertFalse(starts(case))

    def test_pending_active_and_revoking_external_member_prevent_group(self):
        with harness() as case:
            token = p.control(case.socket_path, {
                'action': 'acquire', 'owner': 'forced-group-test', 'requested_mib': 1024,
                'ttl': 30, 'justification': 'CPU scope exclusion test',
                'expected_duration_seconds': 30, 'gpu_uuids': [PAIR[1]],
            })['lease']['token']
            for state in ('pending', 'active', 'revoking'):
                if state == 'active':
                    p.control(case.socket_path, {'action': 'ready', 'token': token})
                if state == 'revoking':
                    p.control(case.socket_path, {'action': 'revoke', 'token': token})
                code, result, _ = capacity(case)
                self.assertEqual(code, 503, result)
                self.assertFalse(starts(case))

    def test_busy_singleton_finishes_without_cancel_before_group_transition(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[PAIR[0]])[0], 200)
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(p.chat, case.proxy_port, p.MODEL, 'busy-before-group',
                                     delay=0.4, gpu_uuids=[PAIR[0]])
                p.wait_until(lambda: bool(p.request_events(case.event_log)), 'native request starts')
                code, result, _ = capacity(case)
                self.assertEqual(code, 503, result)
                self.assertEqual(len(starts(case)), 1)
                self.assertEqual(future.result()[0], 200)
            self.assertEqual(capacity(case)[0], 200)

    def test_incomplete_off_scope_or_cpu_residency_never_publishes_ready_group(self):
        for settings in (
            {'runner_gpu_scope': [PAIR[0]]},
            {'runner_gpu_scope': GPUS},
            {'runner_vram_by_gpu': {PAIR[0]: 1024, PAIR[1]: 0}},
            {'cpu_only': True}, {'partial_gpu_residency': True},
        ):
            with self.subTest(settings=settings), harness(**settings) as case:
                code, result, _ = capacity(case)
                self.assertEqual(code, 503, result)
                self.assertFalse([lane for lane in p.managed_lanes(case.status())
                                  if lane['state'] == 'ready'])

    def test_current_group_member_loss_blocks_proof_and_certified_dispatch(self):
        with harness() as case:
            self.assertEqual(capacity(case)[0], 200)
            code, proof, _ = capacity(case, warm_admission_proof=True)
            self.assertEqual(code, 200, proof)
            pid = starts(case)[0]['pid']
            usage = pathlib.Path(case.gpu_usage_dir) / f'{pid}.csv'
            usage.write_text(f'{PAIR[0]},1024\n')
            code, result, _ = capacity(case, warm_admission_proof=True)
            self.assertEqual(code, 503, result)
            self.assertIn('native_gpu_placement_mismatch', result['warm_preflight_causes'])
            self.assertEqual(capacity(case)[0], 503)
            response = send(case, proof['warm_admission'])
            self.assertEqual(response[0], 503, response)
            self.assertFalse(p.request_events(case.event_log))
            self.assertEqual(len(starts(case)), 1)

    def test_real_model_exceeding_exact_group_capacity_is_permanent(self):
        huge = {'name': p.MODEL, 'model': p.MODEL, 'size': 200 * 1024**3,
                'capabilities': ['completion']}
        with harness(tags=[huge]) as case:
            code, result, _ = capacity(case)
            self.assertEqual(code, 422, result)
            self.assertEqual(result['reason_code'], 'model_exceeds_gpu_capacity')
            self.assertIs(result['retryable'], False)
            self.assertFalse(starts(case))

    def test_group_parallel_slots_are_not_multiple_overlapping_groups(self):
        with harness(instance_parallel=2) as case:
            code, result, _ = capacity(case, parallel=2)
            self.assertEqual(code, 200, result)
            self.assertEqual(len(result['lanes']), 1)
            self.assertEqual(result['admitted_parallel'], 2)
            self.assertEqual(capacity(case, parallel=2, warm_admission_proof=True)[0], 200)
            self.assertEqual(capacity(case, parallel=3)[0], 422)
            self.assertEqual(capacity(case, parallel=3, warm_admission_proof=True)[0], 422)
            self.assertEqual(len(starts(case)), 1)

    def test_cold_group_proof_is_read_only(self):
        with harness() as case:
            self.assertEqual(capacity(case, warm_admission_proof=True)[0], 503)
            self.assertFalse(starts(case))


if __name__ == '__main__':
    unittest.main()
