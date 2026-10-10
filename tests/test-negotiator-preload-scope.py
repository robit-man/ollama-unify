#!/usr/bin/env python3
"""CPU-only actual HTTP/proc controls for the pre-load server scope fence."""
import importlib.machinery
import importlib.util
import pathlib
import sys
import tempfile
import unittest

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
loader = importlib.machinery.SourceFileLoader(
    'preload_scope_pool', str(pathlib.Path(__file__).with_name('test-negotiator-pool.py')))
spec = importlib.util.spec_from_loader(loader.name, loader)
p = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = p
loader.exec_module(p)
PAIR = ['GPU-large-0', 'GPU-large-1']
CAPACITY = '/.well-known/ollama-unify-gpu-negotiator/capacity'
SECRET = 'forged-environment-secret-must-never-be-serialized'


def wrapper(path, mode):
    # Fault only the observed proc metadata of this real fake-native child,
    # after its real version handshake. Never modify the child environment.
    path.write_text('''#!/usr/bin/env python3
import builtins, importlib.machinery, importlib.util, io, logging, pathlib, sys
loader=importlib.machinery.SourceFileLoader('preload_negotiator', %r)
spec=importlib.util.spec_from_loader(loader.name,loader)
n=importlib.util.module_from_spec(spec);sys.modules[loader.name]=n;loader.exec_module(n)
mode=%r
original_open=builtins.open; original_popen=n.subprocess.Popen; native=n.backend_json_at
children=set(); ready=False; reuse=False
def popen(*args,**kwargs):
    process=original_popen(*args,**kwargs)
    if kwargs.get('start_new_session'): children.add(process.pid)
    return process
def observed_open(path,*args,**kwargs):
    global reuse
    parts=pathlib.Path(path).parts if isinstance(path,(str,bytes,pathlib.Path)) else ()
    child=len(parts)==4 and parts[1]=='proc' and parts[2].isdigit() and int(parts[2]) in children
    if ready and child and parts[3]=='environ':
        if mode=='unreadable': raise PermissionError('private environment detail must stay private')
        with original_open(path,*args,**kwargs) as f: raw=f.read()
        entries=raw.split(b'\\0'); key=b'CUDA_VISIBLE_DEVICES='
        value=next(entry[len(key):] for entry in entries if entry.startswith(key))
        if mode=='missing': entries=[entry for entry in entries if not entry.startswith(key)]
        elif mode=='duplicate_key': entries.append(key+b'%s')
        elif mode in ('wrong','duplicate','extra','reversed'):
            values=value.split(b',')
            changed={'wrong':b'GPU-unselected','duplicate':values[0]+b','+values[0],
                     'extra':value+b',GPU-unselected','reversed':b','.join(reversed(values))}[mode]
            entries=[key+changed if entry.startswith(key) else entry for entry in entries]
        elif mode=='reused_during_read': reuse=True
        return io.BytesIO(b'\\0'.join(entries))
    if ready and child and parts[3]=='stat' and (mode in ('reused_pid','changed_pgrp') or reuse):
        with original_open(path,*args,**kwargs) as f: raw=f.read()
        prefix,tail=raw.rsplit(')',1); fields=tail.split()
        field=2 if mode=='changed_pgrp' else 19
        fields[field]=str(int(fields[field])+1)
        return io.StringIO(prefix+') '+' '.join(fields))
    return original_open(path,*args,**kwargs)
def backend(*args,**kwargs):
    global ready
    result=native(*args,**kwargs)
    if args[2:4]==('GET','/api/version'): ready=True
    return result
n.subprocess.Popen=popen;builtins.open=observed_open;n.backend_json_at=backend
logging.basicConfig(filename=%r,level=logging.INFO)
raise SystemExit(n.main())
''' % (str(pathlib.Path(HELPER).resolve()), mode, SECRET, str(path) + '.log'))
    path.chmod(0o755)


class PreloadScopeHTTPTests(unittest.TestCase):
    def run_capacity(self, mode, scope, grouped=False):
        with tempfile.TemporaryDirectory(prefix='preload-scope-cpu-') as directory:
            helper = pathlib.Path(directory) / 'observed-helper.py'
            wrapper(helper, mode)
            with p.PoolHarness(str(helper), FIXTURE_BIN, auto_model_context=True,
                               max_servers=3, selected_gpus=PAIR,
                               runner_vram_by_gpu={gpu: 1024 for gpu in PAIR}) as case:
                body = {'model': p.MODEL, 'parallel': 1, 'endpoint': '/api/chat', 'gpu_uuids': scope}
                if grouped:
                    body['placement'] = 'exclusive_group'
                result = p.http_json(case.proxy_port, 'POST', CAPACITY, body)
                events = p.events(case.event_log)
                loads = [event for event in events if event['kind'] == 'request']
                starts = [event for event in events if event['kind'] == 'start']
                if mode == 'valid':
                    self.assertEqual(result[0], 200, result)
                    self.assertEqual(len(loads), 1)
                    self.assertEqual(loads[0]['options']['num_predict'], 0)
                    self.assertEqual(starts[0]['gpu'], ','.join(scope))
                else:
                    self.assertEqual(result[0], 503, result)
                    self.assertEqual(result[1]['reason_code'], 'gpu_placement_unverified')
                    self.assertEqual(loads, [], 'No zero-token or generating model POST is permitted')
                    self.assertEqual(p.managed_lanes(case.status()), [])
                    self.assertEqual(len(starts), 1)
                    self.assertFalse(pathlib.Path(f"/proc/{starts[0]['pid']}").exists())
                    self.assertNotIn(SECRET, str(result))
                    self.assertNotIn('private environment detail', str(result))
                    self.assertNotIn(SECRET, pathlib.Path(str(helper) + '.log').read_text())
                return result

    def test_actual_singleton_scope_passes_before_zero_token_load(self):
        self.run_capacity('valid', PAIR[:1])

    def test_actual_exact_ordered_group_passes_before_zero_token_load(self):
        self.run_capacity('valid', list(reversed(PAIR)), grouped=True)

    def test_wrong_missing_duplicate_extra_and_reversed_scope_send_no_load(self):
        for mode in ('wrong', 'missing', 'duplicate', 'duplicate_key', 'extra', 'reversed'):
            with self.subTest(mode=mode):
                self.run_capacity(mode, PAIR, grouped=True)

    def test_unreadable_actual_scope_sends_no_load_and_cleans_owned_group(self):
        self.run_capacity('unreadable', PAIR, grouped=True)

    def test_reused_pid_or_changed_group_before_or_during_read_sends_no_load(self):
        for mode in ('reused_pid', 'changed_pgrp', 'reused_during_read'):
            with self.subTest(mode=mode):
                self.run_capacity(mode, PAIR, grouped=True)


if __name__ == '__main__':
    unittest.main()
