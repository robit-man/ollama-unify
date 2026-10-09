#!/usr/bin/env python3
"""CPU-only HTTP/proc regressions for conditional warm-lane admission."""
import concurrent.futures
import copy
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/warm-admission-config'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/warm-admission-leases'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/warm-admission-policy'


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


n = load('warm_admission_negotiator', HELPER)
p = load('warm_admission_pool_fixture', pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
MODEL, DIGEST = p.MODEL, p.MODEL_DIGEST
GPUS = ['GPU-large-0', 'GPU-large-1']
HEADER = 'X-Ollama-Unify-Warm-Admission'
CAPACITY = '/.well-known/ollama-unify-gpu-negotiator/capacity'


def harness(**kwargs):
    return p.PoolHarness(HELPER, FIXTURE_BIN, auto_model_context=True,
                         selected_gpus=GPUS, max_servers=3, **kwargs)


def proof(h, parallel=1):
    return p.http_json(h.proxy_port, 'POST', CAPACITY, {
        'model': MODEL, 'parallel': parallel, 'endpoint': '/api/chat',
        'gpu_uuids': GPUS, 'warm_admission_proof': True,
    })


def send(h, certificate, request_id='warm-request', logical_id=None, delay=0,
         extra=None, body=None, gpu_uuids=GPUS):
    headers = {'X-Ollama-Unify-Workload-Class': 'foreground'}
    if gpu_uuids is not None:
        headers['X-Ollama-Unify-GPU-UUIDs'] = ','.join(gpu_uuids)
    if certificate is not None:
        headers[HEADER] = json.dumps(certificate)
    if logical_id:
        headers['X-Ollama-Unify-Logical-Request-Id'] = logical_id
    headers.update(extra or {})
    payload = body if body is not None else {
        'model': MODEL, 'stream': False, 'mock_request_id': request_id,
        'mock_delay': delay, 'options': {'num_ctx': 262144},
    }
    return p.http_json(h.proxy_port, 'POST', '/api/chat', payload,
                       timeout=8, extra_headers=headers)


def resume(h, logical_id, certificate=None, extra=None):
    headers = {'X-Ollama-Unify-Logical-Request-Id': logical_id,
               'X-Ollama-Unify-Resume-Request': 'true'}
    if certificate is not None:
        headers[HEADER] = json.dumps(certificate)
    headers.update(extra or {})
    return p.http_json(h.proxy_port, 'POST', '/api/chat', None,
                       timeout=8, extra_headers=headers)


class WarmAdmissionHTTPTests(unittest.TestCase):
    def prebackend_fault(self, validation_call):
        # Run the actual public proxy/queue/dispatch code. Only the native PS
        # observation is faulted once, after the request has an active owner.
        with tempfile.TemporaryDirectory(prefix='warm-prebackend-fault-') as directory:
            wrapper = pathlib.Path(directory) / 'fault-helper.py'
            wrapper.write_text('''#!/usr/bin/env python3
import importlib.machinery, importlib.util, sys
loader = importlib.machinery.SourceFileLoader('fault_negotiator', %r)
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)
validate = n.Broker._validate_warm_admission_locked
native_json = n.backend_json_at
calls = 0
fault = False
def inspect(*args, **kwargs):
    if fault and args[2:4] == ('GET', '/api/ps'):
        return {'models': []}
    return native_json(*args, **kwargs)
def validate_once(self, *args, **kwargs):
    global calls, fault
    calls += 1
    fault = calls == %d
    try:
        return validate(self, *args, **kwargs)
    finally:
        fault = False
n.backend_json_at = inspect
n.Broker._validate_warm_admission_locked = validate_once
raise SystemExit(n.main())
''' % (str(pathlib.Path(HELPER).resolve()), validation_call))
            wrapper.chmod(0o755)
            with p.PoolHarness(str(wrapper), FIXTURE_BIN, auto_model_context=True,
                               selected_gpus=GPUS, max_servers=3) as h:
                certificate = self.prepare(h)['warm_admission']
                logical = 'warm:prebackend-fault'
                body = {'model': MODEL, 'stream': False,
                        'mock_request_id': 'original-exact-body', 'options': {'num_ctx': 262144}}
                failed = send(h, certificate, logical_id=logical, body=body)
                self.stale(failed, retained=False)
                self.assertEqual(failed[1]['warm_preflight_phase'],
                                 'post_admission_validation' if validation_call == 3 else 'backend_dispatch_validation')
                self.assertEqual(failed[1]['warm_preflight_causes'], ['native_ps_inventory_mismatch'])
                self.assertEqual(failed[1]['cause_reason_code'], 'native_ps_inventory_mismatch')
                self.assertNotIn('resume_ttl_ms', failed[1])
                state = h.status()
                self.assertEqual(state['active_requests'], 0)
                self.assertEqual(state['parallel_pool']['request_lifecycle']['tracked'], 0)
                self.assertEqual(state['parallel_pool']['queue']['retained_request_bytes'], 0)
                self.assertEqual(p.request_events(h.event_log), [])
                missing = resume(h, logical, certificate)
                self.assertEqual(missing[0], 409, missing)
                self.assertEqual(missing[1]['reason_code'], 'logical_request_not_found')
                self.assertEqual(h.capacity(MODEL, 2, '/api/chat', GPUS)[0], 200)
                fresh = proof(h, 2)[1]['warm_admission']
                self.assertNotEqual(fresh, certificate)
                retried = send(h, fresh, logical_id=logical, body=body)
                self.assertEqual(retried[0], 200, retried)
                replay = resume(h, logical, fresh)
                self.assertEqual(replay[0], 200, replay)
                self.assertEqual(len(p.request_events(h.event_log)), 1)
                self.assertEqual(p.request_events(h.event_log)[0]['request_id'], 'original-exact-body')

    def test_post_prepare_warm_failure_releases_only_proven_unstarted_owner(self):
        self.prebackend_fault(3)

    def test_pre_dispatch_warm_failure_releases_only_proven_unstarted_owner(self):
        self.prebackend_fault(4)

    def prepare(self, h, parallel=1):
        status, result, _ = h.capacity(MODEL, parallel, '/api/chat', GPUS)
        self.assertEqual(status, 200, result)
        status, result, _ = proof(h, parallel)
        self.assertEqual(status, 200, result)
        return result

    def stale(self, response, retained=False):
        self.assertEqual(response[0], 503, response)
        self.assertEqual(response[1]['reason_code'], 'warm_preflight_required')
        self.assertIs(response[1]['admission_retained'], retained)
        self.assertIs(response[1]['backend_started'], False)

    def test_proof_only_does_not_load_and_exposes_actual_process_identity(self):
        with harness() as h:
            self.stale(proof(h))
            self.assertEqual(p.managed_lanes(h.status()), [])
            self.assertEqual(p.events(h.event_log), [])
            result = self.prepare(h, 2)
            self.assertEqual(len(result['warm_admission']['lanes']), 2)
            self.assertEqual(result['warm_admission']['model_digest'], DIGEST)
            self.assertEqual(result['warm_admission']['context_length'], 262144)
            starts = [e for e in p.events(h.event_log) if e['kind'] == 'start']
            for lane in result['warm_admission_lanes']:
                self.assertIn(lane['server_process']['pid'], [e['pid'] for e in starts])
                self.assertIn(lane['server_process'], lane['runtime_processes'])
                self.assertGreater(lane['server_process']['start_time_ticks'], 0)
                self.assertEqual(lane['native_ps']['size_vram'], lane['native_ps']['size'])
                self.assertNotIn('expires_at', lane['native_ps'])
                self.assertNotIn('port', lane)
            self.assertEqual(proof(h, 2)[1]['warm_admission'], result['warm_admission'])
            self.assertEqual(len([e for e in p.events(h.event_log) if e['kind'] == 'start']), 2)

    def test_idle_retirement_rejects_before_retention_then_same_id_can_recertify(self):
        with harness(idle_timeout=1) as h:
            old = self.prepare(h)['warm_admission']
            p.wait_until(lambda: not p.managed_lanes(h.status()), 'ordinary idle retirement', timeout=5)
            before = h.status()['parallel_pool']['queue']['enqueued_total']
            self.stale(send(h, old, logical_id='warm:never-admitted'))
            self.assertEqual(h.status()['parallel_pool']['queue']['enqueued_total'], before)
            self.assertEqual(p.request_events(h.event_log), [])
            self.stale(proof(h))
            fresh = self.prepare(h)['warm_admission']
            self.assertNotEqual(old['lanes'], fresh['lanes'])
            self.assertEqual(send(h, fresh, logical_id='warm:never-admitted')[0], 200)
            self.assertEqual(len(p.request_events(h.event_log)), 1)

    def test_partial_retirement_uses_surviving_certified_lane_only(self):
        with harness() as h:
            result = self.prepare(h, 2)
            certificate = result['warm_admission']
            retired = certificate['lanes'][1]['id']
            p.control(h.socket_path, {'action': 'stop_lane', 'lane_id': retired})
            response = send(h, certificate)
            self.assertEqual(response[0], 200, response)
            self.assertEqual(response[2]['X-Ollama-Unify-Lane'], certificate['lanes'][0]['id'])
            self.assertEqual(len(p.managed_lanes(h.status())), 1)

    def test_busy_certified_lane_queues_without_growth_or_new_lane_selection(self):
        with harness(resume_ttl=3) as h:
            certificate = self.prepare(h)['warm_admission']
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(send, h, certificate, 'busy-certified', 'warm:busy', 1.1)
                p.wait_until(lambda: len(p.request_events(h.event_log)) == 1, 'certified active request')
                queued = send(h, certificate, 'queued-certified', 'warm:queued', extra={
                    'X-Ollama-Unify-Admission-Wait-Ms': '100'})
                self.assertEqual(queued[0], 503, queued)
                self.assertIs(queued[1]['admission_retained'], True)
                self.assertEqual(len(p.managed_lanes(h.status())), 1)
                # An unrelated explicit capacity expansion cannot widen the
                # already-retained conditional job's eligible lane set.
                self.assertEqual(h.capacity(MODEL, 2, '/api/chat', GPUS)[0], 200)
                fresh = proof(h, 2)[1]['warm_admission']
                self.assertNotEqual(fresh, certificate)
                self.assertEqual(resume(h, 'warm:queued', fresh)[0], 409)
                self.assertEqual(send(h, None, 'queued-certified', 'warm:queued')[0], 409)
                self.assertEqual(first.result()[0], 200)
                done = resume(h, 'warm:queued')
                self.assertEqual(done[0], 200, done)
                self.assertEqual(done[2]['X-Ollama-Unify-Lane'], certificate['lanes'][0]['id'])
                self.assertEqual(len(p.request_events(h.event_log)), 2)

    def test_completed_replay_survives_retirement_but_cannot_change_certificate(self):
        with harness(completed_ttl=10) as h:
            certificate = self.prepare(h)['warm_admission']
            first = send(h, certificate, logical_id='warm:completed')
            self.assertEqual(first[0], 200, first)
            p.control(h.socket_path, {'action': 'stop_lane', 'lane_id': certificate['lanes'][0]['id']})
            replay = send(h, certificate, logical_id='warm:completed')
            self.assertEqual(replay[0], 200, replay)
            self.assertEqual(replay[1], first[1])
            self.assertEqual(replay[2]['X-Ollama-Unify-Response-Replayed'], 'true')
            self.assertEqual(resume(h, 'warm:completed')[0], 200)
            self.assertEqual(resume(h, 'warm:completed', extra={
                'X-Ollama-Unify-GPU-UUIDs': GPUS[0]})[0], 409)
            changed = copy.deepcopy(certificate)
            changed['lanes'][0]['generation_id'] = 'f' * 64
            self.assertEqual(resume(h, 'warm:completed', changed)[0], 409)
            self.assertEqual(send(h, None, logical_id='warm:completed')[0], 409)
            self.assertEqual(len(p.request_events(h.event_log)), 1)

    def test_completed_expiry_and_eviction_retain_certificate_in_existing_tombstone(self):
        for mode in ['expired', 'evicted']:
            with self.subTest(mode=mode), harness(completed_ttl=1 if mode == 'expired' else 10,
                                                 completed_max_entries=1) as h:
                certificate = self.prepare(h)['warm_admission']
                self.assertEqual(send(h, certificate, logical_id='warm:old-completion')[0], 200)
                if mode == 'expired':
                    p.wait_until(lambda: h.status()['parallel_pool']['completed_responses']['entries'] == 0,
                                 'completed cache expiry', timeout=2)
                else:
                    self.assertEqual(send(h, certificate, 'replacement', 'warm:replacement')[0], 200)
                changed = copy.deepcopy(certificate)
                changed['lanes'][0]['generation_id'] = 'f' * 64
                self.assertEqual(send(h, None, logical_id='warm:old-completion')[0], 409)
                self.assertEqual(send(h, changed, logical_id='warm:old-completion')[0], 409)
                self.assertEqual(send(h, certificate, logical_id='warm:old-completion')[0], 200)

    def test_full_body_new_certificate_can_start_after_ordinary_tombstone_expiry(self):
        with harness(completed_ttl=0.3) as h:
            certificate = self.prepare(h)['warm_admission']
            self.assertEqual(send(h, certificate, logical_id='warm:post-ttl')[0], 200)
            p.wait_until(lambda: h.status()['parallel_pool']['completed_responses']['entries'] == 0,
                         'cache expiry before new certificate')
            self.assertEqual(h.capacity(MODEL, 2, '/api/chat', GPUS)[0], 200)
            new_certificate = proof(h, 2)[1]['warm_admission']
            self.assertNotEqual(certificate, new_certificate)
            time.sleep(0.35)
            # No body-free resume has run: full-body admission itself must
            # honor the ordinary bounded tombstone lifetime.
            response = send(h, new_certificate, logical_id='warm:post-ttl')
            self.assertEqual(response[0], 200, response)

    def test_stale_tombstone_does_not_advertise_safe_recertification(self):
        with harness(completed_ttl=1) as h:
            certificate = self.prepare(h)['warm_admission']
            self.assertEqual(send(h, certificate, logical_id='warm:owned-tombstone')[0], 200)
            p.wait_until(lambda: h.status()['parallel_pool']['completed_responses']['entries'] == 0,
                         'completed record becomes owned tombstone')
            p.control(h.socket_path, {'action': 'stop_lane', 'lane_id': certificate['lanes'][0]['id']})
            self.stale(send(h, certificate, logical_id='warm:owned-tombstone'), retained=True)

    def test_yield_timeout_cannot_remove_original_retained_certificate(self):
        with harness() as h:
            certificate = self.prepare(h)['warm_admission']
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(send, h, certificate, 'yield-active', None, 0.8)
                p.wait_until(lambda: len(p.request_events(h.event_log)) == 1, 'active lane for yield')
                yielded = send(h, certificate, 'yield-pending', 'warm:yield', extra={
                    'X-Ollama-Unify-Admission-Wait-Ms': '100', 'X-Ollama-Unify-Queue-Policy': 'yield'})
                self.assertEqual(yielded[0], 503, yielded)
                self.assertEqual(send(h, None, 'yield-pending', 'warm:yield')[0], 409)
                self.assertEqual(first.result()[0], 200)

    def test_prediction_options_remain_unchanged_on_native_wire(self):
        with harness() as h:
            certificate = self.prepare(h)['warm_admission']
            options = {'num_ctx': 262144, 'num_gpu': -1, 'num_predict': 257,
                       'temperature': 0.15, 'top_p': 0.8, 'seed': 7, 'stop': ['fixture-stop']}
            response = send(h, certificate, body={
                'model': MODEL, 'mock_request_id': 'sampling-preserved', 'options': options})
            self.assertEqual(response[0], 200, response)
            self.assertEqual(p.request_events(h.event_log)[0]['options'], options)

    def test_boot_generation_context_digest_and_scope_mismatches_never_dispatch(self):
        with harness() as h:
            certificate = self.prepare(h)['warm_admission']
            for key, value in [('broker_instance_id', 'f' * 64), ('model_digest', 'f' * 64),
                               ('context_length', 8192), ('gpu_uuids', list(reversed(GPUS)))]:
                changed = copy.deepcopy(certificate)
                changed[key] = value
                body = {'model': MODEL, 'options': {'num_ctx': changed['context_length']}}
                self.stale(send(h, changed, logical_id='warm:mismatch:' + key, body=body))
            changed = copy.deepcopy(certificate)
            changed['lanes'][0]['generation_id'] = 'f' * 64
            self.stale(send(h, changed, logical_id='warm:mismatch:generation'))
            self.stale(send(h, certificate, gpu_uuids=None))
            self.assertEqual(p.request_events(h.event_log), [])
            self.assertEqual(h.status()['parallel_pool']['queue']['enqueued_total'], 0)
            self.assertEqual(send(h, certificate)[0], 200)

    def test_malformed_and_runtime_reload_options_rejected_before_claim(self):
        with harness() as h:
            certificate = self.prepare(h)['warm_admission']
            for extra in [{HEADER: '{}'}, {HEADER: 'null'}, {HEADER: ''}]:
                response = send(h, None, extra=extra)
                self.assertEqual(response[0], 400, response)
            for changes in [*[{'keep_alive': value} for value in (0, '0s', '0m', '0ms', '0.0s', -1, '30m')],
                            {'options': {'num_ctx': 8192}}, {'options': {'num_gpu': 0}},
                            {'options': {'main_gpu': 0}}, {'runner': 'foreign-variant'},
                            *[{'options': {key: True}} for key in (
                                'num_batch', 'num_thread', 'low_vram', 'use_mmap', 'use_mlock',
                                'f16_kv', 'vocab_only', 'logits_all', 'draft_num_predict',
                                'future_loader_option')]]:
                body = {'model': MODEL, 'mock_request_id': 'must-not-dispatch', **changes}
                response = send(h, certificate, logical_id='warm:invalid', body=body)
                self.assertEqual(response[0], 400, response)
            self.assertEqual(h.status()['parallel_pool']['queue']['enqueued_total'], 0)
            self.assertEqual(send(h, certificate, logical_id='warm:invalid')[0], 200)

    def test_partial_gpu_residency_cannot_be_certified_singleton(self):
        with harness(partial_gpu_residency=True) as h:
            # Legacy loading policy is unchanged; only issuing the opt-in
            # full-residency certificate rejects this partial offload.
            self.assertEqual(h.capacity(MODEL, 1, '/api/chat', GPUS)[0], 200)
            self.stale(proof(h))
            self.assertEqual(p.request_events(h.event_log), [])

    def test_embedding_only_native_context_can_use_warm_background_yield_without_growth(self):
        # Observe the exact synthetic native input in the existing CPU backend;
        # the broker, loader, scheduler and proof path remain production code.
        with tempfile.TemporaryDirectory(prefix='warm-embed-fixture-') as directory:
            fixture_bin = pathlib.Path(directory) / 'bin'
            shutil.copytree(FIXTURE_BIN, fixture_bin)
            backend = fixture_bin / 'ollama'
            original = backend.read_text()
            marker = 'options=payload.get("options") or {}, keep_alive=payload.get("keep_alive"))'
            self.assertEqual(original.count(marker), 1)
            backend.write_text(original.replace(marker, marker[:-1] + ', observed_input=payload.get("input"))'))
            tag = {'name': p.EMBED_MODEL, 'model': p.EMBED_MODEL, 'size': 1024 ** 3,
                   'digest': DIGEST, 'capabilities': ['embedding']}
            with p.PoolHarness(HELPER, str(fixture_bin), tags=[tag], max_servers=3,
                               selected_gpus=GPUS, auto_model_context=True,
                               runner_context_length=2048, max_context=8192) as h:
                setup = h.capacity(p.EMBED_MODEL, 1, '/api/embed', GPUS)
                self.assertEqual(setup[0], 200, setup)
                self.assertIsNone(setup[1]['lanes'][0]['context_profile'])
                issued = p.http_json(h.proxy_port, 'POST', CAPACITY, {
                    'model': p.EMBED_MODEL, 'parallel': 1, 'endpoint': '/api/embed',
                    'gpu_uuids': GPUS, 'warm_admission_proof': True,
                })
                self.assertEqual(issued[0], 200, issued)
                certificate = issued[1]['warm_admission']
                self.assertEqual(certificate['model'], p.EMBED_MODEL)
                self.assertEqual(certificate['model_digest'], DIGEST)
                self.assertEqual(certificate['context_length'], 2048)
                inputs = ['synthetic embedding alpha', 'synthetic embedding beta']
                headers = {
                    HEADER: json.dumps(certificate), 'X-Ollama-Unify-GPU-UUIDs': ','.join(GPUS),
                    'X-Ollama-Unify-Workload-Class': 'background', 'X-Ollama-Unify-Queue-Policy': 'yield',
                }
                for options in [None, {'num_ctx': 2048}]:
                    body = {'model': p.EMBED_MODEL, 'input': inputs,
                            'mock_request_id': 'warm-embedding-input'}
                    if options is not None:
                        body['options'] = options
                    response = p.http_json(h.proxy_port, 'POST', '/api/embed', body, extra_headers=headers)
                    self.assertEqual(response[0], 200, response)
                    self.assertEqual(response[2]['X-Ollama-Unify-Workload-Class'], 'background')
                    self.assertEqual(response[2]['X-Ollama-Unify-Queue-Policy'], 'yield')
                records = p.request_events(h.event_log)
                self.assertEqual(len(records), 2)
                for record in records:
                    self.assertEqual(record['path'], '/api/embed')
                    self.assertEqual(record['observed_input'], inputs)
                    self.assertEqual(record['options'], {'num_ctx': 2048, 'num_gpu': -1})
                conflict = p.http_json(h.proxy_port, 'POST', '/api/embed', {
                    'model': p.EMBED_MODEL, 'input': inputs, 'options': {'num_ctx': 8192},
                }, extra_headers={**headers, 'X-Ollama-Unify-Logical-Request-Id': 'warm:embed-conflict'})
                self.assertEqual(conflict[0], 400, conflict)
                self.assertEqual(h.status()['parallel_pool']['queue']['enqueued_total'], 2)
                self.assertEqual(len(p.request_events(h.event_log)), 2)
                self.assertEqual(len(p.managed_lanes(h.status())), 1)
                self.assertEqual(len([event for event in p.events(h.event_log) if event['kind'] == 'start']), 1)

    def test_embedding_metadata_failure_after_admission_releases_owned_request_before_backend(self):
        tag = {'name': p.EMBED_MODEL, 'model': p.EMBED_MODEL, 'size': 1024 ** 3,
               'digest': DIGEST, 'capabilities': ['embedding']}
        calls = {'enabled': False, 'tags': 0}
        original_get = p.Handler.do_GET

        def change_metadata(handler):
            if calls['enabled'] and handler.path == '/api/tags':
                calls['tags'] += 1
                # Original-body clamp + two warm admission inspections read
                # five tag snapshots. Refuse the preparation reopen next.
                if calls['tags'] >= 6:
                    handler.send_json({'models': [{**tag, 'capabilities': ['embedding', 'completion']}]})
                    return
            original_get(handler)

        with mock.patch.object(p.Handler, 'do_GET', change_metadata), \
             p.PoolHarness(HELPER, FIXTURE_BIN, tags=[tag], max_servers=3,
                           selected_gpus=GPUS, auto_model_context=True,
                           runner_context_length=2048, max_context=8192) as h:
            self.assertEqual(h.capacity(p.EMBED_MODEL, 1, '/api/embed', GPUS)[0], 200)
            issued = p.http_json(h.proxy_port, 'POST', CAPACITY, {
                'model': p.EMBED_MODEL, 'parallel': 1, 'endpoint': '/api/embed',
                'gpu_uuids': GPUS, 'warm_admission_proof': True})
            self.assertEqual(issued[0], 200, issued)
            calls['enabled'] = True
            response = p.http_json(h.proxy_port, 'POST', '/api/embed', {
                'model': p.EMBED_MODEL, 'input': ['must-not-reach-native-backend'],
                'mock_request_id': 'metadata-changed-after-admission',
            }, extra_headers={HEADER: json.dumps(issued[1]['warm_admission']),
                'X-Ollama-Unify-GPU-UUIDs': ','.join(GPUS),
                'X-Ollama-Unify-Logical-Request-Id': 'warm:embedding-after-admission'})
            self.stale(response, retained=False)
            state = h.status()
            self.assertEqual(state['parallel_pool']['queue']['admitted_total'], 1)
            self.assertEqual(state['active_requests'], 0)
            self.assertEqual(state['parallel_pool']['request_lifecycle']['tracked'], 0)
            self.assertEqual(p.request_events(h.event_log), [])
            missing = resume(h, 'warm:embedding-after-admission')
            self.assertEqual(missing[0], 409, missing)
            self.assertEqual(missing[1]['reason_code'], 'logical_request_not_found')

    def test_broker_restart_and_reused_lane_id_cannot_reuse_old_certificate(self):
        with harness() as first:
            original = self.prepare(first)['warm_admission']
        with harness() as second:
            current = self.prepare(second)['warm_admission']
            self.assertEqual(original['lanes'][0]['id'], current['lanes'][0]['id'])
            self.assertNotEqual(original['broker_instance_id'], current['broker_instance_id'])
            self.stale(send(second, original, logical_id='warm:restart'))
            self.assertEqual(send(second, current, logical_id='warm:restart')[0], 200)


class WarmEmbeddingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen(['sleep', '30'], start_new_session=True)
        self.addCleanup(self.stop_process)
        self.tag = {'name': p.EMBED_MODEL, 'digest': DIGEST, 'capabilities': ['embedding']}
        self.show = {'capabilities': ['embedding']}
        self.row = {'name': p.EMBED_MODEL, 'digest': DIGEST, 'context_length': 2048,
                    'size': 4096, 'size_vram': 4096}
        self.lane = n.Lane('lane-1', 'managed', '127.0.0.1', 1, GPUS[0], p.EMBED_MODEL, 1, 1,
                           time.time(), time.time(), self.process, gpu_uuids=(GPUS[0],))
        self.broker = n.Broker()
        self.broker.lanes[self.lane.lane_id] = self.lane
        for patcher in [
            mock.patch.object(n, 'POOL_ENABLED', True),
            mock.patch.object(n, 'HARD_MAX_CONTEXT', 0),
            mock.patch.object(n, 'effective_model_context_profile', return_value=None),
            mock.patch.object(n, 'backend_json_at', side_effect=lambda *a, **kw: {'models': [copy.deepcopy(self.row)]}),
            mock.patch.object(n, 'backend_json', side_effect=self.metadata),
            mock.patch.object(self.broker, '_ollama_blocked_gpus_locked', return_value=set()),
            mock.patch.object(self.broker, '_policy_constraint_locked', side_effect=lambda _model, scope: scope),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def stop_process(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, 15)
        self.process.wait(timeout=3)

    def metadata(self, method, path, *args, **kwargs):
        if (method, path) == ('GET', '/api/tags'):
            return {'models': [copy.deepcopy(self.tag)]}
        if (method, path) == ('POST', '/api/show'):
            return copy.deepcopy(self.show)
        raise AssertionError((method, path))

    def issue(self):
        return self.broker.warm_admission_proof(p.EMBED_MODEL, 1, tuple(GPUS))

    def test_missing_tag_capabilities_are_reopened_from_show_without_completion_profile(self):
        self.tag.pop('capabilities')
        issued = self.issue()
        self.assertIsNone(self.lane.context_profile)
        self.assertEqual(issued['warm_admission']['context_length'], 2048)
        self.assertEqual(issued['warm_admission']['model_digest'], DIGEST)

    def test_unprofiled_completion_mixed_and_unknown_capabilities_cannot_certify(self):
        for capabilities in [['completion'], ['embedding', 'completion'], [], ['unknown']]:
            with self.subTest(capabilities=capabilities):
                self.tag['capabilities'] = capabilities
                self.show['capabilities'] = capabilities
                with self.assertRaises(n.WarmPreflightRequired):
                    self.issue()
        self.tag.pop('capabilities')
        self.show.pop('capabilities')
        with self.assertRaises(n.WarmPreflightRequired):
            self.issue()

    def test_installed_and_native_digests_must_match(self):
        self.tag['digest'] = 'b' * 64
        with self.assertRaises(n.WarmPreflightRequired):
            self.issue()

    def test_tag_replacement_during_show_cannot_certify_old_artifact(self):
        before = {'models': [{'name': p.EMBED_MODEL, 'digest': DIGEST}]}
        after = {'models': [{'name': p.EMBED_MODEL, 'digest': 'b' * 64}]}
        with mock.patch.object(n, 'backend_json', side_effect=[before, self.show, after]):
            with self.assertRaises(n.WarmPreflightRequired):
                self.issue()

    def test_observed_embedding_context_must_be_positive_and_within_hard_limit(self):
        for context in [0, -1, True, '2048', 2 ** 31]:
            with self.subTest(context=context):
                self.row['context_length'] = context
                with self.assertRaises(n.WarmPreflightRequired):
                    self.issue()
        self.row['context_length'] = 2048
        with mock.patch.object(n, 'HARD_MAX_CONTEXT', 1024):
            with self.assertRaises(n.WarmPreflightRequired):
                self.issue()

    def test_native_context_change_invalidates_issued_embedding_certificate(self):
        certificate = n.parse_warm_admission(json.dumps(self.issue()['warm_admission']))
        self.row['context_length'] = 4096
        with self.broker.cv, self.assertRaises(n.WarmPreflightRequired):
            self.broker._validate_warm_admission_locked(certificate, p.EMBED_MODEL, tuple(GPUS))


class WarmOwnerLifecycleTests(unittest.TestCase):
    def owned(self, broker=None, certificate=None):
        broker = broker or n.Broker()
        lane = n.Lane('lane-owned', 'managed', '127.0.0.1', 1, GPUS[0], MODEL, 1, 1,
                      time.time(), time.time(), gpu_uuids=(GPUS[0],))
        certificate = certificate or json.dumps({'gpu_uuids': GPUS})
        with broker.cv:
            active = broker._register_active_request_locked(lane, 'same-request', 'warm:same-logical')
            active.warm_admission = certificate
            broker.logical_in_flight['warm:same-logical'] = ('exact-body-fingerprint', 'same-request', 7)
        admission = n.Admission(lane, 'same-request', 'warm:same-logical', 'exact-body-fingerprint',
                                0, 1, 7, warm_admission=certificate, active_request=active)
        return broker, lane, active, admission

    def test_proven_release_fences_late_dispatch_watcher_and_finally_from_replacement(self):
        broker, old_lane, old, admission = self.owned()
        failure = n.WarmPreflightRequired(phase='post_admission_validation', causes=('native_ps_unavailable',))
        released = broker.reject_warm_before_backend(admission, failure)
        self.assertIs(released, failure)
        self.assertIs(released.admission_retained, False)
        self.assertEqual(old_lane.in_flight, 0)
        self.assertEqual(broker.active_requests, 0)
        self.assertNotIn(admission.logical_request_id, broker.logical_tombstones)
        # Reuse even the physical request ID adversarially. Identity must be
        # the active record object, not an ID that a stale handler can retake.
        _, new_lane, replacement, new_admission = self.owned(broker)
        before = vars(replacement).copy()
        with mock.patch.object(broker, '_validate_warm_admission_locked', return_value={'lane-owned'}) as validate:
            with self.assertRaises(n.WarmAdmissionOwnershipUncertain):
                broker.request_backend_started(admission, object())
            validate.assert_not_called()
            self.assertFalse(broker.renew_request_activity(admission, 'stale-update'))
            broker.bind_active_request_client(admission, 'stale-client')
            broker.request_client_detached(admission)
            broker.note_backend_failure(admission, 'stale-failure')
            self.assertFalse(broker.request_backend_complete(admission))
            self.assertIsNone(broker.active_request_terminal_lock(admission))
            self.assertEqual(broker.active_request_cancel_reason(admission), '')
            broker.proxy_exit(admission, MODEL, False)
            self.assertEqual(vars(replacement), before)
            self.assertIs(broker.active_request_records['same-request'], replacement)
            self.assertEqual(new_lane.in_flight, 1)
            self.assertTrue(broker.request_backend_started(new_admission, object()))
            broker.proxy_exit(admission, MODEL, False)
            self.assertIs(broker.active_request_records['same-request'], replacement)
            self.assertTrue(replacement.backend_started)

    def test_missing_owner_is_uncertain_not_a_safe_recertification_claim(self):
        broker, lane, active, admission = self.owned()
        with broker.cv:
            broker._release_active_request_locked(admission.request_id, MODEL, False)
        failure = broker.reject_warm_before_backend(admission, n.WarmPreflightRequired())
        self.assertIsInstance(failure, n.WarmAdmissionOwnershipUncertain)
        self.assertFalse(failure.retryable)
        self.assertEqual(lane.in_flight, 0)
        with self.assertRaises(n.WarmAdmissionOwnershipUncertain):
            broker.request_backend_started(admission, object())

    def test_cancelled_detached_started_and_conflicting_owners_are_never_released(self):
        cases = ['cancelled', 'detached', 'started', 'backend-object', 'completed',
                 'foreign-record', 'changed-certificate', 'changed-fingerprint',
                 'waiter', 'replay', 'tombstone']
        for case in cases:
            with self.subTest(case=case):
                broker, lane, active, admission = self.owned()
                if case == 'cancelled': active.cancel_requested_at = time.monotonic()
                elif case == 'detached': active.detached_at = time.monotonic()
                elif case == 'started': active.backend_started = True
                elif case == 'backend-object': active.backend = object()
                elif case == 'completed': active.backend_completed = True
                elif case == 'foreign-record':
                    admission = n.Admission(lane, admission.request_id, admission.logical_request_id,
                        admission.request_fingerprint, 0, 1, 7, warm_admission=admission.warm_admission)
                elif case == 'changed-certificate': active.warm_admission = 'foreign-certificate'
                elif case == 'changed-fingerprint':
                    broker.logical_in_flight[admission.logical_request_id] = ('foreign-body', admission.request_id, 7)
                elif case == 'waiter':
                    broker.waiters.append(types.SimpleNamespace(request_id='other-request',
                        logical_request_id=admission.logical_request_id))
                elif case == 'replay': broker.completed_responses[admission.logical_request_id] = object()
                elif case == 'tombstone': broker.logical_tombstones[admission.logical_request_id] = object()
                before = vars(active).copy()
                failure = broker.reject_warm_before_backend(admission, n.WarmPreflightRequired())
                self.assertIsInstance(failure, n.WarmAdmissionOwnershipUncertain)
                self.assertFalse(failure.retryable)
                self.assertEqual(failure.status, 409)
                self.assertIs(broker.active_request_records[admission.request_id], active)
                self.assertEqual(vars(active), before)
                self.assertEqual(lane.in_flight, 1)
                self.assertEqual(broker.active_requests, 1)

    def test_pre_dispatch_cancel_or_already_started_never_revalidates_or_overwrites_transport(self):
        for state in ['cancelled', 'detached', 'started', 'backend-object']:
            with self.subTest(state=state):
                broker, lane, active, admission = self.owned()
                if state == 'cancelled': active.cancel_requested_at = time.monotonic()
                elif state == 'detached': active.detached_at = time.monotonic()
                elif state == 'started': active.backend_started = True
                else: active.backend = object()
                before = vars(active).copy()
                with mock.patch.object(broker, '_validate_warm_admission_locked') as validate:
                    with self.assertRaises(n.WarmAdmissionOwnershipUncertain):
                        broker.request_backend_started(admission, object())
                    validate.assert_not_called()
                self.assertEqual(vars(active), before)
                self.assertEqual(lane.in_flight, 1)

    def test_diagnostics_serialize_only_closed_safe_cause_and_phase_values(self):
        handler = object.__new__(n.ProxyHandler)
        responses = []
        handler._send_json = lambda status, body, headers: responses.append((status, body, headers))
        failure = n.WarmPreflightRequired(phase='secret-phase-token',
            causes=('secret-body-token', 'native_ps_unavailable', {'secret': 'token'}))
        handler._send_capacity_failure(failure)
        body = responses[0][1]
        self.assertEqual(body['cause_reason_code'], 'native_ps_unavailable')
        self.assertEqual(body['warm_preflight_phase'], 'validation')
        self.assertEqual(body['warm_preflight_causes'], ['native_ps_unavailable'])
        self.assertNotIn('secret', json.dumps(responses))
        _, _, _, admission = self.owned()
        handler._send_capacity_failure(n.WarmAdmissionOwnershipUncertain(admission, failure))
        status, uncertain, _ = responses[-1]
        self.assertEqual(status, 409)
        self.assertFalse(uncertain['retryable'])
        self.assertEqual(uncertain['reason_code'], 'warm_admission_ownership_uncertain')
        self.assertNotIn('backend_started', uncertain)
        self.assertNotIn('admission_retained', uncertain)
        self.assertNotIn('resume_ttl_ms', uncertain)


class WarmNativeIdentityTests(unittest.TestCase):
    def test_proof_issuer_reports_only_actually_differing_identity_fields(self):
        for field, value, expected in [('context_length', 4096, 'context_mismatch'),
                                       ('digest', 'f' * 64, 'model_digest_mismatch')]:
            with self.subTest(field=field):
                broker = n.Broker()
                lanes = [n.Lane('lane-9' + str(i), 'managed', '127.0.0.1', i + 1,
                    gpu, p.EMBED_MODEL, 1, 1, time.time(), time.time(), gpu_uuids=(gpu,))
                    for i, gpu in enumerate(GPUS)]
                broker.lanes.update({lane.lane_id: lane for lane in lanes})
                base = {'name': p.EMBED_MODEL, 'digest': DIGEST, 'context_length': 2048,
                        'size': 4096, 'size_vram': 4096}
                def inspected(lane):
                    return {'server_process': {'pid': 1, 'start_time_ticks': 1},
                            'runtime_processes': [{'pid': 1, 'start_time_ticks': 1}],
                            'native_ps': base if lane is lanes[0] else {**base, field: value}}
                with mock.patch.object(n, 'POOL_ENABLED', True), \
                     mock.patch.object(broker, '_ollama_blocked_gpus_locked', return_value=set()), \
                     mock.patch.object(broker, '_policy_constraint_locked', side_effect=lambda _model, scope: scope), \
                     mock.patch.object(broker, '_warm_runtime_identity_locked', side_effect=inspected):
                    with self.assertRaises(n.WarmPreflightRequired) as failure:
                        broker.warm_admission_proof(p.EMBED_MODEL, 2, tuple(GPUS))
                    self.assertEqual(failure.exception.warm_preflight_phase, 'proof_issuance')
                    self.assertEqual(failure.exception.warm_preflight_causes, (expected,))

    def test_strict_native_failures_keep_closed_predicate_and_call_stage_diagnostics(self):
        profile = {'model_digest': DIGEST, 'context_length': 262144, 'extra_vram_mib': 1}
        process = types.SimpleNamespace(pid=43210)
        lane = n.Lane('lane-99', 'managed', '127.0.0.1', 1, GPUS[0], MODEL, 1, 1,
                      time.time(), time.time(), process, gpu_uuids=(GPUS[0],), context_profile=profile)
        row = {'name': MODEL, 'digest': DIGEST, 'context_length': 262144,
               'size': 4096, 'size_vram': 4096}
        identity = [{'pid': process.pid, 'start_time_ticks': 1234}]
        broker = n.Broker()
        broker.lanes[lane.lane_id] = lane
        with mock.patch.object(n, 'POOL_ENABLED', True), \
             mock.patch.object(n, 'effective_model_context_profile', return_value=profile), \
             mock.patch.object(n, 'process_group_alive', return_value=True), \
             mock.patch.object(n, 'process_group_identity', return_value=identity), \
             mock.patch.object(n, 'backend_json_at', return_value={'models': [row]}) as native, \
             mock.patch.object(broker, '_ollama_blocked_gpus_locked', return_value=set()), \
             mock.patch.object(broker, '_policy_constraint_locked', side_effect=lambda _model, scope: scope):
            issued = broker.warm_admission_proof(MODEL, 1, tuple(GPUS))
            certificate = n.parse_warm_admission(json.dumps(issued['warm_admission']))
            cases = [
                (TimeoutError('secret native exception'), None, 'native_ps_unavailable'),
                (None, {'models': []}, 'native_ps_inventory_mismatch'),
                (None, {'models': [{**row, 'name': 'foreign-model'}]}, 'native_model_mismatch'),
                (None, {'models': [{**row, 'digest': 'f' * 64}]}, 'native_digest_mismatch'),
                (None, {'models': [{**row, 'context_length': 8192}]}, 'native_context_mismatch'),
                (None, {'models': [{**row, 'size_vram': 0}]}, 'native_gpu_residency_mismatch'),
            ]
            for error, result, cause in cases:
                with self.subTest(cause=cause):
                    native.side_effect = error
                    native.return_value = result
                    with self.assertRaises(n.WarmPreflightRequired) as failure:
                        broker._validate_warm_admission_locked(certificate, MODEL, tuple(GPUS),
                                                              phase='post_admission_validation')
                    self.assertEqual(failure.exception.warm_preflight_causes, (cause,))
                    self.assertEqual(failure.exception.warm_preflight_phase, 'post_admission_validation')
                    self.assertNotIn('secret', str(failure.exception))

    def test_real_process_group_includes_native_children_and_detects_replacement(self):
        process = subprocess.Popen([sys.executable, '-c',
            'import subprocess,time; subprocess.Popen(["sleep","30"]); time.sleep(30)'],
            start_new_session=True)
        self.addCleanup(lambda: os.killpg(process.pid, 15) if process.poll() is None else None)
        before = p.wait_until(lambda: (items if len(items := n.process_group_identity(process)) == 2 else None),
                              'native child identity')
        self.assertEqual(len(before), 2)
        self.assertTrue(all(item['start_time_ticks'] > 0 for item in before))
        child = next(item for item in before if item['pid'] != process.pid)
        os.kill(child['pid'], 15)
        after = p.wait_until(lambda: (items if len(items := n.process_group_identity(process)) == 1 else None),
                             'native child disappearance')
        self.assertNotEqual(before, after)
        os.killpg(process.pid, 15)
        process.wait(timeout=3)

    def test_native_child_replacement_invalidates_certificate_before_backend_start(self):
        process = subprocess.Popen([sys.executable, '-c',
            'import subprocess,time; subprocess.Popen(["sleep","30"]); time.sleep(30)'],
            start_new_session=True)
        try:
            before = p.wait_until(lambda: (items if len(items := n.process_group_identity(process)) == 2 else None),
                                  'native child before certification')
            profile = {'model_digest': DIGEST, 'context_length': 262144, 'extra_vram_mib': 1}
            lane = n.Lane('lane-1', 'managed', '127.0.0.1', 1, GPUS[0], MODEL, 1, 1,
                          time.time(), time.time(), process, gpu_uuids=(GPUS[0],), context_profile=profile)
            row = {'name': MODEL, 'digest': DIGEST, 'context_length': 262144,
                   'size': 4096, 'size_vram': 4096, 'expires_at': 'volatile'}
            broker = n.Broker()
            broker.lanes['lane-1'] = lane
            with mock.patch.object(n, 'POOL_ENABLED', True), \
                 mock.patch.object(n, 'effective_model_context_profile', return_value=profile), \
                 mock.patch.object(n, 'backend_json_at', return_value={'models': [row]}), \
                 mock.patch.object(broker, '_ollama_blocked_gpus_locked', return_value=set()), \
                 mock.patch.object(broker, '_policy_constraint_locked', side_effect=lambda _model, scope: scope):
                issued = broker.warm_admission_proof(MODEL, 1, tuple(GPUS))
                certificate = n.parse_warm_admission(json.dumps(issued['warm_admission']))
                with broker.cv:
                    active = broker._register_active_request_locked(lane, 'actual-dispatch', 'warm:native-child')
                    active.warm_admission = certificate
                    broker.logical_in_flight['warm:native-child'] = ('fingerprint', 'actual-dispatch', 1)
                admission = n.Admission(lane, 'actual-dispatch', 'warm:native-child', 'fingerprint', 0, 1, 1,
                                        warm_admission=certificate, active_request=active)
                child = next(item for item in before if item['pid'] != process.pid)
                os.kill(child['pid'], 15)
                p.wait_until(lambda: len(n.process_group_identity(process)) == 1, 'native child exit')
                with self.assertRaises(n.WarmPreflightRequired) as failure:
                    broker.request_backend_started(admission, object())
                self.assertIs(failure.exception.admission_retained, False)
                self.assertEqual(failure.exception.warm_preflight_phase, 'backend_dispatch_validation')
                self.assertEqual(failure.exception.warm_preflight_causes, ('runtime_identity_changed',))
                self.assertNotIn('actual-dispatch', broker.active_request_records)
                self.assertNotIn('warm:native-child', broker.logical_in_flight)
                self.assertNotIn('warm:native-child', broker.logical_tombstones)
                refreshed = broker.warm_admission_proof(MODEL, 1, tuple(GPUS))
                self.assertNotEqual(issued['warm_admission']['lanes'], refreshed['warm_admission']['lanes'])
        finally:
            os.killpg(process.pid, 15)
            process.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
