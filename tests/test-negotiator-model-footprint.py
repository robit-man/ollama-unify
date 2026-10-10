#!/usr/bin/env python3
"""CPU-only exact manifest/native graph footprint and conservative fallback controls."""
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/footprint-config'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/footprint-leases'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/footprint-policy'
HELPER = sys.argv.pop(1)
loader = importlib.machinery.SourceFileLoader('footprint_negotiator', HELPER)
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)

MODEL = 'fixture-owner/multimodal:exact'
AUX = 'application/vnd.robit.ollama.omni.bundle.v1+gguf'
PREFIX = 'application/vnd.ollama.image.'
MIB = 1024 ** 2


class ModelFootprintTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='footprint-cpu-')
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.path = self.root / 'manifests/registry.ollama.ai/fixture-owner/multimodal/exact'
        self.path.parent.mkdir(parents=True)
        (self.root / 'blobs').mkdir()
        self.layers = [self.layer('model', 4 * MIB), self.layer('projector', MIB),
                       self.layer('template', 13), self.layer(AUX, 30 * MIB)]
        self.manifest = {'schemaVersion': 2, 'mediaType': 'application/vnd.docker.distribution.manifest.v2+json',
                         'config': {'mediaType': 'application/vnd.docker.container.image.v1+json',
                                    'digest': 'sha256:' + 'c' * 64, 'size': 237},
                         'layers': self.layers}
        self.tag = {'name': MODEL, 'model': MODEL, 'capabilities': ['completion']}
        self.show = {'modelfile': self.modelfile(), 'capabilities': ['completion']}
        self.calls = []
        self.during_show = None
        self.save_manifest()
        for key, value in (('OLLAMA_MODELS', str(self.root)), ('POOL_MODEL_OVERHEAD_PERCENT', 110),
                           ('POOL_VRAM_RESERVE_MIB', 1024), ('POOL_INSTANCE_PARALLEL', 2)):
            patch = mock.patch.object(n, key, value)
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(n, 'backend_json', side_effect=self.backend)
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(n, 'resolve_model_context_profile', return_value={
            'extra_vram_mib': 8192, 'context_length': 262144, 'model_digest': self.tag['digest']})
        patch.start(); self.addCleanup(patch.stop)
        self.broker = n.Broker()

    def layer(self, kind, size):
        media = kind if '/' in kind else PREFIX + kind
        digest = hashlib.sha256((media + str(size)).encode()).hexdigest()
        path = self.root / 'blobs' / ('sha256-' + digest)
        # Sparse files exercise the actual stat path without large fixtures.
        with path.open('wb') as stream:
            stream.truncate(size)
        return {'mediaType': media, 'digest': 'sha256:' + digest, 'size': size}

    def modelfile(self):
        return '\n'.join('FROM "' + str(self.root / 'blobs' / layer['digest'].replace(':', '-')) + '"'
                         for layer in self.layers if layer['mediaType'] in
                         {PREFIX + 'model', PREFIX + 'projector', PREFIX + 'tensor'}) + '\n'

    def save_manifest(self):
        raw = json.dumps(self.manifest, separators=(',', ':')).encode()
        self.path.write_bytes(raw)
        self.tag['digest'] = hashlib.sha256(raw).hexdigest()
        self.tag['size'] = sum(layer['size'] for layer in self.layers) + self.manifest['config']['size']

    def backend(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, payload))
        if (method, path) == ('GET', '/api/tags'):
            return {'models': [copy.deepcopy(self.tag)]}
        if (method, path) == ('POST', '/api/show'):
            self.assertEqual(payload, {'model': MODEL})
            result = copy.deepcopy(self.show)
            if self.during_show:
                self.during_show()
            return result
        self.fail('No model loading or inference is allowed in footprint tests')

    def expected(self, subtract=0):
        model_mib = math.ceil((self.tag['size'] - subtract) / MIB)
        return math.ceil(model_mib * 110 / 100) + 1024 + 8192 * 2

    def required(self):
        return self.broker._model_profile(MODEL)[0]

    def test_exact_native_model_projector_graph_excludes_only_known_auxiliary(self):
        self.assertEqual(self.required(), self.expected(self.layers[-1]['size']))
        self.assertEqual([x[:2] for x in self.calls],
                         [('GET', '/api/tags'), ('POST', '/api/show'), ('GET', '/api/tags')])

    def test_adapters_and_tensor_weights_are_all_reserved(self):
        self.layers += [self.layer('adapter', 2 * MIB), self.layer('tensor', MIB)]
        self.save_manifest()
        self.show['modelfile'] = self.modelfile() + 'ADAPTER ' + str(
            self.root / 'blobs' / self.layers[-2]['digest'].replace(':', '-'))
        self.assertEqual(self.required(), self.expected(self.layers[3]['size']))

    def test_operator_context_floor_parallel_and_overhead_are_unchanged(self):
        # A larger explicit context reserve is not replaced with low observed VRAM.
        self.assertEqual(self.required(), math.ceil(6 * 1.1) + 1024 + 8192 * 2)

    def test_unknown_media_type_preserves_full_aggregate(self):
        self.layers.append(self.layer('application/vnd.future.runtime.weight', 2 * MIB))
        self.save_manifest()
        self.assertEqual(self.required(), self.expected())

    def test_auxiliary_that_native_modelfile_references_is_never_excluded(self):
        self.show['modelfile'] += 'FROM ' + str(self.root / 'blobs' / self.layers[-1]['digest'].replace(':', '-'))
        self.assertEqual(self.required(), self.expected())

    def test_draft_runtime_role_is_not_mistaken_for_unloaded_auxiliary(self):
        self.show['modelfile'] += 'DRAFT ' + str(self.root / 'blobs' / self.layers[-1]['digest'].replace(':', '-'))
        self.assertEqual(self.required(), self.expected())

    def test_missing_extra_duplicate_or_foreign_native_blob_falls_back(self):
        valid = self.show['modelfile']
        for mode in ('missing', 'extra', 'duplicate', 'foreign', 'relative', 'malformed'):
            with self.subTest(mode=mode):
                lines = valid.splitlines()
                self.show['modelfile'] = {'missing': lines[0], 'extra': valid + 'FROM /bad/sha256-' + 'e' * 64,
                    'duplicate': valid + lines[0], 'foreign': valid.replace(str(self.root), '/foreign'),
                    'relative': valid.replace(str(self.root) + '/', ''), 'malformed': valid + 'ADAPTER "unterminated'}[mode]
                self.assertEqual(self.required(), self.expected())

    def test_changed_manifest_or_tag_during_native_metadata_falls_back(self):
        original = self.path.read_bytes()
        old_digest = self.tag['digest']
        for mode in ('manifest', 'tag'):
            with self.subTest(mode=mode):
                self.path.write_bytes(original); self.tag['digest'] = old_digest
                self.during_show = (lambda: self.path.write_bytes(original + b' ')) if mode == 'manifest' else (
                    lambda: self.tag.update(digest='e' * 64))
                self.assertEqual(self.required(), self.expected())

    def test_changed_runtime_blob_size_during_metadata_falls_back(self):
        path = self.root / 'blobs' / self.layers[0]['digest'].replace(':', '-')
        self.during_show = lambda: path.write_bytes(b'changed')
        self.assertEqual(self.required(), self.expected())

    def test_unavailable_native_metadata_preserves_original_aggregate(self):
        with mock.patch.object(n, 'backend_json', side_effect=lambda method, path, *a, **kw:
                               {'models': [self.tag]} if path == '/api/tags' else (_ for _ in ()).throw(OSError('unavailable'))):
            self.assertEqual(self.required(), self.expected())

    def test_nonobject_native_metadata_envelopes_preserve_aggregate(self):
        original = self.backend
        for endpoint in ('/api/show', '/api/tags'):
            for invalid in (None, [], 'invalid'):
                with self.subTest(endpoint=endpoint, invalid=invalid):
                    tags_calls = 0
                    def malformed(method, path, payload=None, **kwargs):
                        nonlocal tags_calls
                        if path == '/api/tags': tags_calls += 1
                        if path == endpoint and (path == '/api/show' or tags_calls > 1): return invalid
                        return original(method, path, payload, **kwargs)
                    with mock.patch.object(n, 'backend_json', side_effect=malformed):
                        self.assertEqual(self.required(), self.expected())

    def test_blob_and_manifest_symlink_aliases_do_not_authorize_smaller_reservation(self):
        blob = self.root / 'blobs' / self.layers[0]['digest'].replace(':', '-')
        original_blob = blob.with_name('original-model')
        blob.rename(original_blob); blob.symlink_to(original_blob)
        self.assertEqual(self.required(), self.expected())
        blob.unlink(); original_blob.rename(blob)
        original_manifest = self.path.with_name('original-manifest')
        self.path.rename(original_manifest); self.path.symlink_to(original_manifest)
        self.assertEqual(self.required(), self.expected())

    def test_blob_alias_retarget_during_metadata_is_not_hidden_by_old_target(self):
        blob = self.root / 'blobs' / self.layers[0]['digest'].replace(':', '-')
        target = blob.with_name('other-model')
        with target.open('wb') as stream: stream.truncate(self.layers[0]['size'])
        def replace():
            blob.unlink(); blob.symlink_to(target)
        self.during_show = replace
        self.assertEqual(self.required(), self.expected())

    def test_store_and_directory_alias_retarget_preserves_aggregate(self):
        for kind in ('store', 'manifests', 'blobs'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='footprint-alias-') as directory:
                external = pathlib.Path(directory)
                replacement = external / 'replacement'
                replacement.mkdir()
                if kind == 'store':
                    alias = external / 'store'
                    alias.symlink_to(self.root, target_is_directory=True)
                    patch = mock.patch.object(n, 'OLLAMA_MODELS', str(alias))
                    original = None
                else:
                    alias = self.root / kind
                    original = external / 'original'
                    alias.rename(original)
                    alias.symlink_to(original, target_is_directory=True)
                    patch = mock.patch.object(n, 'OLLAMA_MODELS', str(self.root))
                # Native SHOW uses the resolved runtime files. Retargeting only
                # an original directory alias must not hide behind intact files.
                self.show['modelfile'] = '\n'.join('FROM "' + str(
                    (self.root / 'blobs' / layer['digest'].replace(':', '-')).resolve()) + '"'
                    for layer in self.layers if layer['mediaType'] in {PREFIX + 'model', PREFIX + 'projector'})
                def replace():
                    alias.unlink(); alias.symlink_to(replacement, target_is_directory=True)
                self.during_show = replace
                try:
                    with patch:
                        self.assertEqual(self.required(), self.expected())
                finally:
                    alias.unlink()
                    if original is not None: original.rename(alias)
                    self.during_show = None
                    self.show['modelfile'] = self.modelfile()

    def test_missing_local_store_manifest_or_runtime_blob_falls_back(self):
        with mock.patch.object(n, 'OLLAMA_MODELS', ''):
            self.assertEqual(self.required(), self.expected())
        self.path.unlink()
        self.assertEqual(self.required(), self.expected())
        self.save_manifest()
        (self.root / 'blobs' / self.layers[0]['digest'].replace(':', '-')).unlink()
        self.assertEqual(self.required(), self.expected())

    def test_manifest_identity_size_schema_and_duplicate_integrity_fall_back(self):
        original = copy.deepcopy(self.manifest)
        for mode in ('digest', 'schema', 'boolean_size', 'negative_size', 'bad_digest', 'duplicate', 'tag_size', 'bad_config'):
            with self.subTest(mode=mode):
                self.manifest = copy.deepcopy(original); self.layers = self.manifest['layers']
                if mode == 'schema': self.manifest['schemaVersion'] = 3
                if mode == 'boolean_size': self.layers[0]['size'] = True
                if mode == 'negative_size': self.layers[0]['size'] = -1
                if mode == 'bad_digest': self.layers[0]['digest'] = 'sha256:invalid'
                if mode == 'duplicate': self.layers.append(copy.deepcopy(self.layers[0]))
                if mode == 'bad_config': self.manifest['config']['size'] = 40 * MIB
                self.save_manifest()
                if mode == 'digest': self.tag['digest'] = 'f' * 64
                if mode == 'tag_size': self.tag['size'] += MIB
                self.assertEqual(self.required(), self.expected())

    def test_no_auxiliary_keeps_legacy_size_without_extra_metadata_call(self):
        self.layers.pop(); self.save_manifest()
        self.assertEqual(self.required(), self.expected())
        self.assertEqual([x[:2] for x in self.calls], [('GET', '/api/tags')])

    def test_ambiguous_duplicate_json_keys_keep_original_aggregate(self):
        original = self.path.read_bytes()
        raw = original.replace(b'"schemaVersion":2', b'"schemaVersion":3,"schemaVersion":2')
        self.path.write_bytes(raw)
        self.tag['digest'] = hashlib.sha256(raw).hexdigest()
        self.assertEqual(self.required(), self.expected())

    def test_native_tag_size_without_config_bytes_is_also_exact(self):
        self.tag['size'] -= self.manifest['config']['size']
        self.assertEqual(self.required(), self.expected(self.layers[-1]['size']))


if __name__ == '__main__':
    unittest.main()
