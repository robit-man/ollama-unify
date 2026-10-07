#!/usr/bin/env python3
"""CPU-only model-maximum context, identity, and KV admission regressions."""
import copy
import importlib.machinery
import importlib.util
import json
import math
import os
import pathlib
import sys
import unittest
from unittest import mock

os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/context-negotiator-config'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/context-negotiator-leases'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/context-negotiator-policy'
HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
loader = importlib.machinery.SourceFileLoader('model_context_negotiator', HELPER)
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)

MODEL = 'fixture-context:exact'
DIGEST = 'c' * 64
MAXIMUM = 262144
TAG = {'name': MODEL, 'model': MODEL, 'digest': DIGEST,
       'size': 1024**3, 'capabilities': ['completion']}
INFO = {
    'general.architecture': 'llama',
    'llama.context_length': MAXIMUM,
    'llama.block_count': 32,
    'llama.embedding_length': 4096,
    'llama.attention.head_count': 32,
    'llama.attention.head_count_kv': 8,
    'llama.attention.key_length': 128,
    'llama.attention.value_length': 128,
}


class ModelContextTests(unittest.TestCase):
    def setUp(self):
        self.tag = copy.deepcopy(TAG)
        self.info = copy.deepcopy(INFO)
        self.show_capabilities = ['completion']
        self.show_parameters = ''
        self.calls = []
        for name, value in (
            ('MAX_CONTEXT', 8192), ('MODEL_CONTEXT_PROFILES', {}),
            ('AUTO_MODEL_CONTEXT', True), ('HARD_MAX_CONTEXT', 0),
            ('RESOLVED_MODEL_CONTEXT_PROFILES', {}), ('MODEL_CONTEXT_RESOLUTIONS', {}),
            ('POOL_ENABLED', True), ('POOL_INSTANCE_PARALLEL', 2),
            ('POOL_MODEL_OVERHEAD_PERCENT', 100), ('POOL_VRAM_RESERVE_MIB', 1024),
            ('SELECTED_GPUS', ['GPU-context']),
        ):
            patch = mock.patch.object(n, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(n, 'backend_json', side_effect=self.backend)
        patch.start()
        self.addCleanup(patch.stop)
        self.broker = n.Broker()

    def backend(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, payload))
        if method == 'GET' and path == '/api/tags':
            return {'models': [copy.deepcopy(self.tag)]}
        if method == 'POST' and path == '/api/show':
            self.assertEqual(payload['model'], MODEL)
            return {'model_info': copy.deepcopy(self.info),
                    'capabilities': list(self.show_capabilities),
                    'parameters': self.show_parameters}
        self.fail(f'unexpected backend call: {method} {path}')

    def profile(self, context=131072):
        return {'context_length': context, 'extra_vram_mib': 8192,
                'model_digest': DIGEST}

    def test_automatic_verified_maximum_replaces_legacy_global_8192_default(self):
        self.broker._model_profile(MODEL)
        path, payload = self.broker._warm_request(MODEL, {'completion'}, '/api/chat')
        self.assertEqual(path, '/api/generate')
        self.assertEqual(payload['options']['num_ctx'], MAXIMUM)
        self.assertIn(('POST', '/api/show', {'model': MODEL}), self.calls)

    def test_automatic_kv_reservation_covers_q8_cache_for_every_parallel_slot(self):
        required, capabilities = self.broker._model_profile(MODEL)
        # q8_0 stores 32 values plus a two-byte scale per block. K and V
        # each have eight 128-dimensional heads in every attention layer.
        bytes_per_token = 32 * 8 * (128 + 128) * (34 / 32)
        minimum_kv = math.ceil(MAXIMUM * bytes_per_token * 2 / 1024**2)
        self.assertGreaterEqual(required, 1024 + 1024 + minimum_kv)
        self.assertIn('completion', capabilities)

    def test_show_capabilities_resolve_models_whose_tag_metadata_omits_them(self):
        # /api/tags capabilities is optional; /api/show is the metadata source.
        self.tag.pop('capabilities')
        self.broker._model_profile(MODEL)
        _, payload = self.broker._warm_request(MODEL, {'completion'}, '/api/generate')
        self.assertEqual(payload['options']['num_ctx'], MAXIMUM)

    def test_show_embedding_capability_selects_embedding_warmup_when_tags_omit_it(self):
        self.tag.pop('capabilities')
        self.show_capabilities = ['embedding']
        _, capabilities = self.broker._model_profile(MODEL)
        path, _ = self.broker._warm_request(MODEL, capabilities, '')
        self.assertEqual(path, '/api/embed')
        self.assertNotIn(MODEL, n.RESOLVED_MODEL_CONTEXT_PROFILES)

    def test_qwen_hybrid_cache_counts_full_attention_and_recurrent_state(self):
        info = {key.replace('llama.', 'qwen35.'): value for key, value in INFO.items()}
        info['general.architecture'] = 'qwen35'
        info.update({
            'qwen35.full_attention_interval': 4,
            'qwen35.ssm.inner_size': 8192,
            'qwen35.ssm.state_size': 128,
            'qwen35.ssm.group_count': 16,
            'qwen35.ssm.conv_kernel': 4,
        })
        memory = n.estimate_model_context_memory(info, MAXIMUM)
        dense = n.estimate_model_context_memory(INFO, MAXIMUM)
        self.assertEqual(memory['attention_blocks'], 8)
        self.assertEqual(memory['kv_cache_mib'], dense['kv_cache_mib'] // 4)
        self.assertGreater(memory['recurrent_state_mib'], 0)
        self.assertGreaterEqual(memory['extra_vram_mib'],
                                memory['kv_cache_mib'] + memory['recurrent_state_mib'] + 1024)
        info.pop('qwen35.ssm.state_size')
        with self.assertRaises(n.PermanentCapacityError) as raised:
            n.estimate_model_context_memory(info, MAXIMUM)
        self.assertEqual(raised.exception.reason_code, 'model_context_memory_unverified')

    def test_q8_quantization_rounds_key_and_value_dimensions_independently(self):
        self.info['llama.attention.key_length'] = 33
        self.info['llama.attention.value_length'] = 33
        memory = n.estimate_model_context_memory(self.info, MAXIMUM)
        expected = math.ceil(MAXIMUM * 32 * 8 * (2 + 2) * 34 / 1024**2)
        self.assertEqual(memory['kv_cache_mib'], expected)

    def test_unequal_head_dimensions_reserve_actual_f16_fallback_cache(self):
        self.info['llama.attention.key_length'] = 33
        self.info['llama.attention.value_length'] = 31
        memory = n.estimate_model_context_memory(self.info, MAXIMUM)
        expected = math.ceil(MAXIMUM * 32 * 8 * (33 + 31) * 2 / 1024**2)
        self.assertEqual(memory['kv_cache_type'], 'f16')
        self.assertEqual(memory['kv_cache_mib'], expected)

    def test_gemma2_reserves_f16_when_flash_attention_is_disabled(self):
        info = {key.replace('llama.', 'gemma2.'): value for key, value in INFO.items()}
        info['general.architecture'] = 'gemma2'
        memory = n.estimate_model_context_memory(info, MAXIMUM)
        expected = math.ceil(MAXIMUM * 32 * 8 * (128 + 128) * 2 / 1024**2)
        self.assertEqual(memory['kv_cache_type'], 'f16')
        self.assertEqual(memory['kv_cache_mib'], expected)

    def test_pooling_metadata_reserves_f16_even_with_equal_head_dimensions(self):
        self.info['llama.pooling_type'] = 1
        memory = n.estimate_model_context_memory(self.info, MAXIMUM)
        expected = math.ceil(MAXIMUM * 32 * 8 * (128 + 128) * 2 / 1024**2)
        self.assertEqual(memory['kv_cache_type'], 'f16')
        self.assertEqual(memory['kv_cache_mib'], expected)

    def test_manual_context_mode_preserves_an_explicit_digest_bound_profile(self):
        n.AUTO_MODEL_CONTEXT = False
        profile = self.profile()
        n.MODEL_CONTEXT_PROFILES[MODEL] = profile
        required, _ = self.broker._model_profile(MODEL)
        _, payload = self.broker._warm_request(MODEL, {'completion'}, '/api/generate')
        self.assertEqual(payload['options']['num_ctx'], profile['context_length'])
        memory = n.estimate_model_context_memory(INFO, profile['context_length'])
        self.assertEqual(required, 1024 + 1024 +
                         max(profile['extra_vram_mib'], memory['extra_vram_mib']) * 2)

    def test_exact_operator_profile_cannot_understate_known_context_memory(self):
        profile = self.profile(MAXIMUM)
        profile['extra_vram_mib'] = 1
        n.MODEL_CONTEXT_PROFILES[MODEL] = profile
        required, _ = self.broker._model_profile(MODEL)
        minimum = n.estimate_model_context_memory(INFO, MAXIMUM)['extra_vram_mib']
        self.assertEqual(required, 1024 + 1024 + minimum * 2)

    def test_unknown_geometry_accepts_only_unchanged_exact_profile_with_conservative_reserve(self):
        self.info.pop('llama.block_count')
        profile = self.profile(MAXIMUM)
        profile['extra_vram_mib'] = math.ceil(MAXIMUM / 16)
        n.MODEL_CONTEXT_PROFILES[MODEL] = profile
        self.broker._model_profile(MODEL)
        self.assertEqual(n.effective_model_context_profile(MODEL), profile)
        n.RESOLVED_MODEL_CONTEXT_PROFILES.clear()
        n.MODEL_CONTEXT_RESOLUTIONS.clear()
        profile['extra_vram_mib'] = 1
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.broker._model_profile(MODEL)
        self.assertEqual(raised.exception.reason_code, 'model_context_memory_unverified')

    def test_unknown_geometry_cannot_upgrade_an_operator_profile(self):
        self.info.pop('llama.block_count')
        profile = self.profile()
        profile['extra_vram_mib'] = 65536
        n.MODEL_CONTEXT_PROFILES[MODEL] = profile
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.broker._model_profile(MODEL)
        self.assertEqual(raised.exception.reason_code, 'model_context_memory_unverified')

    def test_automatic_mode_upgrades_a_legacy_profile_and_keeps_its_memory_floor(self):
        profile = self.profile()
        n.MODEL_CONTEXT_PROFILES[MODEL] = profile
        required, _ = self.broker._model_profile(MODEL)
        _, payload = self.broker._warm_request(MODEL, {'completion'}, '/api/generate')
        self.assertEqual(payload['options']['num_ctx'], MAXIMUM)
        self.assertTrue(n.MODEL_CONTEXT_RESOLUTIONS[MODEL]['legacy_profile_upgraded'])
        self.assertGreaterEqual(required, 1024 + 1024 + profile['extra_vram_mib'] * 2)
        # A larger verified operator reserve also remains an admission floor.
        profile['extra_vram_mib'] = 65536
        n.RESOLVED_MODEL_CONTEXT_PROFILES.clear()
        n.MODEL_CONTEXT_RESOLUTIONS.clear()
        required, _ = self.broker._model_profile(MODEL)
        self.assertEqual(required, 1024 + 1024 + 65536 * 2)

    def test_explicit_profile_rejects_a_replaced_artifact_digest(self):
        n.MODEL_CONTEXT_PROFILES[MODEL] = self.profile()
        self.tag['digest'] = 'd' * 64
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.broker._model_profile(MODEL)
        self.assertEqual(raised.exception.reason_code, 'model_context_identity_mismatch')

    def test_profile_above_model_metadata_limit_is_rejected(self):
        n.MODEL_CONTEXT_PROFILES[MODEL] = self.profile()
        self.info['llama.context_length'] = 65536
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.broker._model_profile(MODEL)
        self.assertEqual(raised.exception.reason_code, 'model_context_limit_unverified')

    def test_runtime_context_attestation_rejects_a_smaller_resident_context(self):
        n.MODEL_CONTEXT_PROFILES[MODEL] = self.profile()
        self.broker._model_profile(MODEL)
        with self.assertRaises(n.PermanentCapacityError) as raised:
            n.verified_model_context(MODEL, [{'name': MODEL, 'digest': DIGEST, 'context_length': 8192}])
        self.assertEqual(raised.exception.reason_code, 'model_context_runtime_mismatch')

    def test_automatic_resolution_exposes_verified_artifact_maximum_and_memory(self):
        self.broker._model_profile(MODEL)
        resolution = n.MODEL_CONTEXT_RESOLUTIONS[MODEL]
        self.assertEqual(resolution['model_digest'], DIGEST)
        self.assertEqual(resolution['model_max_context'], MAXIMUM)
        self.assertEqual(resolution['context_length'], MAXIMUM)
        self.assertEqual(resolution['kv_cache_type'], 'q8_0')
        self.assertGreater(resolution['kv_cache_mib'], 0)
        self.assertTrue(resolution['source'])
        self.assertFalse(resolution['limit_reason'])

    def test_operator_hard_limit_retains_the_model_maximum_and_constraint_visibility(self):
        n.HARD_MAX_CONTEXT = 65536
        self.broker._model_profile(MODEL)
        resolution = n.MODEL_CONTEXT_RESOLUTIONS[MODEL]
        self.assertEqual(resolution['model_max_context'], MAXIMUM)
        self.assertEqual(resolution['context_length'], 65536)
        self.assertTrue(resolution['limit_reason'])
        _, payload = self.broker._warm_request(MODEL, {'completion'}, '/api/generate')
        self.assertEqual(payload['options']['num_ctx'], 65536)

    def test_native_and_openai_requests_pin_the_same_verified_maximum(self):
        self.broker._model_profile(MODEL)
        for path in ('/api/chat', '/api/generate', '/v1/chat/completions', '/v1/completions', '/v1/responses'):
            with self.subTest(path=path):
                body = json.dumps({'model': MODEL, 'options': {'num_ctx': 8192}}).encode()
                prepared = json.loads(n.clamp_request(path, 'application/json', body))
                self.assertEqual(prepared['options']['num_ctx'], MAXIMUM)

    def test_openai_rejects_model_parameter_that_prevents_the_admitted_context(self):
        self.show_parameters = 'temperature 0.2\nnum_ctx 8192\n'
        self.broker._model_profile(MODEL)
        body = json.dumps({'model': MODEL}).encode()
        native = json.loads(n.clamp_request('/api/chat', 'application/json', body))
        self.assertEqual(native['options']['num_ctx'], MAXIMUM)
        for path in ('/v1/chat/completions', '/v1/completions', '/v1/responses'):
            with self.subTest(path=path), self.assertRaises(n.PermanentCapacityError) as raised:
                n.clamp_request(path, 'application/json', body)
            self.assertEqual(raised.exception.reason_code, 'model_context_openai_mismatch')

    def test_non_object_options_cannot_bypass_context_validation(self):
        self.show_parameters = 'num_ctx 8192\n'
        for path in ('/api/chat', '/api/generate', '/v1/chat/completions', '/v1/completions', '/v1/responses'):
            for options in ([], 'ignored-by-openai', 17, True):
                with self.subTest(path=path, options=options), self.assertRaises(n.PermanentCapacityError) as raised:
                    n.clamp_request(path, 'application/json',
                                    json.dumps({'model': MODEL, 'options': options}).encode())
                self.assertEqual(raised.exception.status, 400)
                self.assertEqual(raised.exception.reason_code, 'invalid_model_options')

    def test_metadata_resolution_rejects_a_tag_replaced_during_show(self):
        backend = self.backend

        def replace_after_show(method, path, payload=None, **kwargs):
            result = backend(method, path, payload, **kwargs)
            if path == '/api/show':
                self.tag['digest'] = 'd' * 64
            return result

        with mock.patch.object(n, 'backend_json', side_effect=replace_after_show):
            with self.assertRaises(n.CapacityError) as raised:
                self.broker._model_profile(MODEL)
        self.assertEqual(raised.exception.reason_code, 'model_context_identity_changed')
        self.assertNotIn(MODEL, n.RESOLVED_MODEL_CONTEXT_PROFILES)

    def test_automatic_runtime_attestation_rejects_a_silently_reduced_context(self):
        self.broker._model_profile(MODEL)
        for context in (8192, MAXIMUM * 2):
            with self.subTest(context=context), self.assertRaises(n.PermanentCapacityError) as raised:
                n.verified_model_context(MODEL, [{'name': MODEL, 'digest': DIGEST, 'context_length': context}])
            self.assertEqual(raised.exception.reason_code, 'model_context_runtime_mismatch')
        self.assertEqual(n.verified_model_context(MODEL, [{'name': MODEL, 'digest': DIGEST, 'context_length': MAXIMUM}]), MAXIMUM)

    def test_runtime_artifact_must_match_the_digest_used_for_memory_admission(self):
        self.broker._model_profile(MODEL)
        for digest in (None, 'd' * 64):
            with self.subTest(digest=digest), self.assertRaises(n.PermanentCapacityError):
                n.verified_model_context(MODEL, [{
                    'name': MODEL, 'digest': digest, 'context_length': MAXIMUM,
                }])

    def test_automatic_replacement_rebinds_maximum_to_the_new_exact_digest(self):
        n.resolve_model_context_profile(MODEL, self.tag)
        self.tag['digest'] = 'd' * 64
        self.info['llama.context_length'] = 65536
        n.resolve_model_context_profile(MODEL, self.tag)
        resolution = n.MODEL_CONTEXT_RESOLUTIONS[MODEL]
        self.assertEqual(resolution['model_digest'], self.tag['digest'])
        self.assertEqual(resolution['context_length'], 65536)

    def resident_lane(self):
        self.broker._model_profile(MODEL)
        now = n.time.time()
        profile = copy.deepcopy(n.effective_model_context_profile(MODEL))
        lane = n.Lane('old-artifact', 'managed', '127.0.0.1', 1, 'GPU-context',
                      MODEL, 2, 40000, now, now, process=mock.Mock(pid=999999),
                      resolved_context_length=MAXIMUM, context_profile=profile)
        self.broker.lanes[lane.lane_id] = lane
        return lane, profile

    def replace_artifact(self):
        self.tag['digest'] = 'd' * 64
        self.info['llama.context_length'] = 65536
        n.resolve_model_context_profile(MODEL, self.tag)

    def test_resident_lane_keeps_its_admitted_profile_and_cannot_route_after_retag(self):
        lane, profile = self.resident_lane()
        with mock.patch.object(n, 'process_group_alive', return_value=True), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ):
            self.assertIs(self.broker._select_lane_locked(MODEL, True), lane)
            self.replace_artifact()
            self.assertIsNone(self.broker._select_lane_locked(MODEL, True))
            self.assertEqual(lane.public_summary()['context_profile'], profile)
        self.assertEqual(lane.reserved_mib, 40000)

    def test_retained_body_uses_selected_lane_context_instead_of_stale_body_context(self):
        lane, _ = self.resident_lane()
        for path in ('/api/chat', '/api/generate'):
            for stale_context in (8192, MAXIMUM * 2):
                with self.subTest(path=path, stale_context=stale_context):
                    prepared = json.loads(self.broker.prepare_managed_body(lane, path,
                        json.dumps({'model': MODEL, 'options': {'num_ctx': stale_context}}).encode()))
                    self.assertEqual(prepared['options']['num_ctx'], MAXIMUM)
                    self.assertEqual(prepared['keep_alive'], -1)

    def test_body_is_not_forwarded_if_selected_lane_identity_changes_after_admission(self):
        lane, _ = self.resident_lane()
        self.replace_artifact()
        body = json.dumps({'model': MODEL, 'options': {'num_ctx': MAXIMUM}}).encode()
        for path in ('/api/chat', '/api/generate', '/v1/chat/completions', '/v1/responses'):
            with self.subTest(path=path), self.assertRaises(n.CapacityError) as raised:
                self.broker.prepare_managed_body(lane, path, body)
            self.assertEqual(raised.exception.reason_code, 'model_context_identity_changed')
            self.assertTrue(raised.exception.retryable)

    def test_retained_openai_body_rechecks_captured_model_context_compatibility(self):
        lane, _ = self.resident_lane()
        lane.openai_context_compatible = False
        body = json.dumps({'model': MODEL}).encode()
        for path in ('/v1/chat/completions', '/v1/completions', '/v1/responses'):
            with self.subTest(path=path), self.assertRaises(n.PermanentCapacityError) as raised:
                self.broker.prepare_managed_body(lane, path, body)
            self.assertEqual(raised.exception.reason_code, 'model_context_openai_mismatch')

    def test_busy_retagged_lane_retains_reservation_and_is_never_stopped_or_reused(self):
        lane, profile = self.resident_lane()
        lane.in_flight = 1
        self.replace_artifact()
        with mock.patch.object(n, 'process_group_alive', return_value=True), mock.patch.object(
            n, 'require_gpu_health'
        ), mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            self.broker, '_stop_lanes'
        ) as stop, mock.patch.object(self.broker, '_spawn_lane') as spawn:
            with self.assertRaises(n.CapacityError) as raised:
                self.broker.ensure_capacity(MODEL, 1, gpu_uuids=('GPU-context',))
        self.assertEqual(raised.exception.reason_code, 'model_context_identity_changed')
        self.assertTrue(raised.exception.retryable)
        self.assertIs(self.broker.lanes[lane.lane_id], lane)
        self.assertEqual(lane.context_profile, profile)
        self.assertEqual(lane.in_flight, 1)
        self.assertEqual(lane.reserved_mib, 40000)
        stop.assert_not_called()
        spawn.assert_not_called()

    def test_profile_change_after_admission_blocks_spawn_before_any_process_starts(self):
        required, capabilities = self.broker._model_profile(MODEL)
        admitted = copy.deepcopy(n.effective_model_context_profile(MODEL))
        self.replace_artifact()
        with mock.patch.object(n, 'require_gpu_health'), mock.patch.object(
            self.broker, '_require_safe_gpu_transition'
        ), mock.patch.object(n.os, 'access', return_value=True), mock.patch.object(
            self.broker, '_available_port_locked', return_value=12345
        ), mock.patch.object(n.subprocess, 'Popen') as start:
            with self.assertRaises(n.CapacityError) as raised:
                self.broker._spawn_lane(MODEL, 'GPU-context', required, capabilities, '',
                                        expected_profile=admitted)
        self.assertEqual(raised.exception.reason_code, 'model_context_identity_changed')
        self.assertTrue(raised.exception.retryable)
        start.assert_not_called()

    def test_idle_retagged_lane_must_stop_before_replacement_starts(self):
        lane, _ = self.resident_lane()
        self.replace_artifact()
        order = []

        def stop_old(lanes, reason):
            self.assertEqual(lanes, [lane])
            order.append('stop-old')
            self.broker.lanes.pop(lane.lane_id)
            return []

        def spawn_new(*args, **kwargs):
            self.assertEqual(order, ['stop-old'])
            order.append('start-new')
            now = n.time.time()
            replacement = n.Lane('new-artifact', 'managed', '127.0.0.1', 2, 'GPU-context',
                MODEL, 2, args[2], now, now, process=mock.Mock(pid=999998),
                resolved_context_length=65536,
                context_profile=copy.deepcopy(n.effective_model_context_profile(MODEL)))
            self.broker.lanes[replacement.lane_id] = replacement
            return replacement

        devices = [{'uuid': 'GPU-context', 'total_mib': 65536, 'free_mib': 65536}]
        with mock.patch.object(n, 'process_group_alive', return_value=True), mock.patch.object(
            n, 'require_gpu_health'
        ), mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            n, 'gpu_snapshot', return_value=devices
        ), mock.patch.object(n, 'host_memory_snapshot', return_value={}), mock.patch.object(
            self.broker, '_placement_devices', return_value=devices
        ), mock.patch.object(self.broker, '_stop_lanes', side_effect=stop_old), mock.patch.object(
            self.broker, '_spawn_lane', side_effect=spawn_new
        ):
            result = self.broker.ensure_capacity(MODEL, 1, gpu_uuids=('GPU-context',))
        self.assertEqual(order, ['stop-old', 'start-new'])
        self.assertEqual(result['admitted_parallel'], 2)
        self.assertEqual(result['lanes'][0]['context_profile']['model_digest'], 'd' * 64)

    def test_missing_kv_geometry_cannot_authorize_unbounded_maximum(self):
        self.info.pop('llama.block_count')
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.broker._model_profile(MODEL)
        self.assertEqual(raised.exception.reason_code, 'model_context_memory_unverified')
        self.assertNotIn(MODEL, n.RESOLVED_MODEL_CONTEXT_PROFILES)

    def test_missing_artifact_digest_cannot_bind_automatic_context(self):
        self.tag.pop('digest')
        with self.assertRaises(n.PermanentCapacityError):
            self.broker._model_profile(MODEL)
        self.assertNotIn(MODEL, n.RESOLVED_MODEL_CONTEXT_PROFILES)

    def test_kv_memory_requirement_blocks_placement_before_any_model_start(self):
        device = {'uuid': 'GPU-context', 'total_mib': 16384, 'free_mib': 16384}
        with mock.patch.object(n, 'gpu_snapshot', return_value=[device]), mock.patch.object(
            n, 'require_gpu_health'
        ), mock.patch.object(n, 'foreign_gpu_usage', return_value={}), mock.patch.object(
            self.broker, '_spawn_lane'
        ) as start:
            with self.assertRaises(n.PermanentCapacityError) as raised:
                self.broker.ensure_capacity(MODEL, 1, gpu_uuids=('GPU-context',))
        self.assertEqual(raised.exception.reason_code, 'model_exceeds_gpu_capacity')
        self.assertIn('requires', str(raised.exception))
        self.assertEqual(n.MODEL_CONTEXT_RESOLUTIONS[MODEL]['model_max_context'], MAXIMUM)
        start.assert_not_called()


@unittest.skipUnless(FIXTURE_BIN, 'requires CPU fixture binary directory')
class ModelContextIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pathlib.Path(__file__).with_name('test-negotiator-pool.py')
        loader = importlib.machinery.SourceFileLoader('context_pool_fixture', str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.pool = importlib.util.module_from_spec(spec)
        sys.modules[loader.name] = cls.pool
        loader.exec_module(cls.pool)

    def test_automatic_context_is_admitted_and_attested_through_managed_http(self):
        p = self.pool
        with p.PoolHarness(HELPER, FIXTURE_BIN, auto_model_context=True, max_context=8192) as case:
            code, discovery, _ = p.http_json(case.proxy_port, 'GET',
                '/.well-known/ollama-unify-gpu-negotiator?model=' + p.MODEL)
            self.assertEqual(code, 200, discovery)
            policy = discovery['context_policy']
            self.assertEqual(policy['managed_context_policy'], 'verified_model_maximum')
            self.assertEqual(policy['legacy_backend_max_context'], 8192)
            self.assertEqual(policy['model_profiles'][p.MODEL]['context_length'], MAXIMUM)
            resolution = policy['model_context_resolutions'][p.MODEL]
            self.assertEqual(resolution['model_max_context'], MAXIMUM)
            self.assertEqual(resolution['model_digest'], p.MODEL_DIGEST)
            self.assertFalse(resolution['limit_reason'])
            self.assertFalse([event for event in p.events(case.event_log) if event['kind'] == 'start'])
            code, capacity, _ = case.capacity(p.MODEL)
            self.assertEqual(code, 200, capacity)
            lane = capacity['lanes'][0]
            self.assertEqual(lane['resolved_context_length'], MAXIMUM)
            self.assertGreater(lane['reserved_mib'], 2048)
            started = next(event for event in p.events(case.event_log) if event['kind'] == 'start')
            self.assertEqual(started['context_length'], MAXIMUM)
            for path in ('/api/chat', '/v1/chat/completions'):
                request_id = 'automatic-' + path
                code, result, _ = p.http_json(case.proxy_port, 'POST', path, {
                    'model': p.MODEL, 'stream': False, 'mock_request_id': request_id,
                    'messages': [{'role': 'user', 'content': 'CPU fixture'}],
                    'options': {'num_ctx': 8192},
                })
                self.assertEqual(code, 200, result)
                request = next(event for event in p.request_events(case.event_log)
                               if event['request_id'] == request_id)
                self.assertEqual(request['options']['num_ctx'], MAXIMUM)

    def test_unknown_resource_geometry_returns_actionable_failure_without_start(self):
        p = self.pool
        with p.PoolHarness(HELPER, FIXTURE_BIN, auto_model_context=True, model_info={
            'general.architecture': 'fixture', 'fixture.context_length': MAXIMUM,
        }) as case:
            code, failure, _ = case.capacity(p.MODEL)
            self.assertEqual(code, 422, failure)
            self.assertEqual(failure['reason_code'], 'model_context_memory_unverified')
            self.assertFalse(failure['retryable'])
            self.assertFalse([event for event in p.events(case.event_log) if event['kind'] == 'start'])

    def test_smaller_runtime_context_never_becomes_a_ready_managed_lane(self):
        p = self.pool
        with p.PoolHarness(HELPER, FIXTURE_BIN, auto_model_context=True,
                           runner_context_length=8192) as case:
            code, failure, _ = case.capacity(p.MODEL)
            self.assertEqual(code, 422, failure)
            self.assertEqual(failure['reason_code'], 'model_context_runtime_mismatch')
            self.assertFalse(failure['retryable'])
            self.assertFalse(p.managed_lanes(case.status()))

    def test_actual_resident_memory_above_admission_budget_never_becomes_ready(self):
        p = self.pool
        with p.PoolHarness(HELPER, FIXTURE_BIN, auto_model_context=True,
                           runner_vram_mib=4096) as case:
            code, failure, _ = case.capacity(p.MODEL)
            self.assertEqual(code, 503, failure)
            self.assertEqual(failure['reason_code'], 'model_context_memory_exceeded')
            self.assertFalse(failure['retryable'])
            self.assertFalse(p.managed_lanes(case.status()))


if __name__ == '__main__':
    unittest.main()
