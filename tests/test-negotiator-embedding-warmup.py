#!/usr/bin/env python3
"""CPU-only zero-work embedding load/relocation through real fake-native HTTP."""
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

HELPER=sys.argv.pop(1)
FIXTURE_BIN=sys.argv.pop(1)
for key in ('CONFIG','LEASE_STATE','MODEL_POLICY_STATE','CACHE_STATE'):
    os.environ['OLLAMA_UNIFY_'+key]='/nonexistent/zero-work-embedding-'+key.lower()


def load(name,path):
    loader=importlib.machinery.SourceFileLoader(name,str(path))
    spec=importlib.util.spec_from_loader(name,loader)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;loader.exec_module(module)
    return module


n=load('zero_work_negotiator',HELPER)
p=load('zero_work_pool',pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
GPUS=['GPU-large-0','GPU-large-1']
EMBED={'name':p.EMBED_MODEL,'model':p.EMBED_MODEL,'size':1024**3,'capabilities':['embedding']}


def requests(case):
    return [e for e in p.events(case.event_log) if e['kind']=='request' and e['model']==p.EMBED_MODEL]


def executions(case):
    return [e for e in p.events(case.event_log) if e['kind']=='embedding_execution']


class ZeroWorkWarmupTests(unittest.TestCase):
    def test_each_embedding_endpoint_uses_empty_input_with_unchanged_identity(self):
        for path in n.EMBEDDING_PATHS:
            with self.subTest(path=path):
                self.assertEqual(n.Broker._warm_request(p.EMBED_MODEL,{'embedding'},path),
                    ('/api/embed',{'model':p.EMBED_MODEL,'input':[],'keep_alive':-1}))
    def test_embedding_only_capability_infers_zero_work_without_endpoint(self):
        self.assertEqual(n.Broker._warm_request(p.EMBED_MODEL,{'embedding'},''),
            ('/api/embed',{'model':p.EMBED_MODEL,'input':[],'keep_alive':-1}))
    def test_explicit_embedding_endpoint_wins_for_dual_capability(self):
        self.assertEqual(n.Broker._warm_request(p.MODEL,{'embedding','completion'},'/api/embed'),
            ('/api/embed',{'model':p.MODEL,'input':[],'keep_alive':-1}))
    def test_completion_and_rerank_contracts_are_unchanged(self):
        path,body=n.Broker._warm_request(p.MODEL,{'completion'},'')
        self.assertEqual(path,'/api/generate');self.assertEqual(body['prompt'],'')
        self.assertEqual(body['options']['num_predict'],0);self.assertEqual(body['keep_alive'],-1)
        self.assertEqual(n.Broker._warm_request('fixture-rerank:latest',{'reranking'},''),
            ('/api/rerank',{'model':'fixture-rerank:latest','query':'warmup','documents':['warmup'],'keep_alive':-1}))
    def assert_cold_zero_work(self,endpoint):
        with p.PoolHarness(HELPER,FIXTURE_BIN,max_servers=2,tags=[EMBED],selected_gpus=GPUS,max_context=8192) as case:
            status,capacity,_=case.capacity(p.EMBED_MODEL,endpoint=endpoint,gpu_uuids=[GPUS[0]])
            self.assertEqual(status,200,capacity)
            seen=requests(case);self.assertEqual(len(seen),1)
            self.assertEqual((seen[0]['path'],seen[0]['input'],seen[0]['keep_alive']),('/api/embed',[],-1))
            self.assertEqual(executions(case),[])
            proof_status,proof,_=p.http_json(case.proxy_port,'POST',
                '/.well-known/ollama-unify-gpu-negotiator/capacity',
                {'model':p.EMBED_MODEL,'parallel':1,'gpu_uuids':[GPUS[0]],'warm_admission_proof':True})
            self.assertEqual(proof_status,200,proof)
            native=proof['warm_admission_lanes'][0]['native_ps']
            self.assertEqual(native['name'],p.EMBED_MODEL)
            self.assertEqual(native['size_vram'],native['size']);self.assertGreater(native['size'],0)
            self.assertEqual(proof['warm_admission_lanes'][0]['gpu_uuids'],[GPUS[0]])
            self.assertEqual(executions(case),[])
            # Only a later genuine inference body executes embedding in the fake
            # native implementation. The setup itself has already been attested.
            self.assertEqual(p.embed(case.proxy_port,p.EMBED_MODEL,'after-placement-proof')[0],200)
            self.assertEqual([e['request_id'] for e in executions(case)],['after-placement-proof'])
    def test_explicit_cold_embedding_only_loads_without_inference(self): self.assert_cold_zero_work('/api/embed')
    def test_inferred_cold_embedding_only_loads_without_inference(self): self.assert_cold_zero_work('')
    def test_two_cache_evacuation_loads_embedding_destination_without_inference(self):
        with p.PoolHarness(HELPER,FIXTURE_BIN,max_servers=4,tags=[p.MODEL,EMBED],selected_gpus=GPUS,max_context=8192) as case:
            originals={}
            for model,endpoint in ((p.MODEL,'/api/chat'),(p.EMBED_MODEL,'/api/embed')):
                status,capacity,_=case.capacity(model,endpoint=endpoint,gpu_uuids=[GPUS[0]])
                self.assertEqual(status,200,capacity);originals[model]=capacity['lanes'][0]['id']
                p.control(case.socket_path,{'action':'set_cache_policy','model':model,'movable':True,'priority':1,'gpu_uuids':[GPUS[1]]})
            result=p.control(case.socket_path,{'action':'acquire','owner':'zero-work-relocation-test',
                'requested_mib':1024,'ttl':30,'gpu_uuids':[GPUS[0]],'priority':10,'evacuation_id':'zero-work-two-cache',
                'justification':'CPU verifies zero-work native loads before two-cache copy-before-retire','expected_duration_seconds':30})
            self.assertEqual(result['lease']['state'],'pending')
            self.assertEqual(result['evacuation']['state'],'lease_pending')
            self.assertEqual(len(result['evacuation']['moves']),2)
            for move in result['evacuation']['moves']:
                self.assertEqual(move['state'],'moved');self.assertEqual(move['source']['lane_id'],originals[move['source']['model']])
                self.assertEqual(move['source']['native'],move['destination']['native'])
                self.assertEqual(move['destination']['gpu_uuids'],[GPUS[1]])
            seen=[e for e in requests(case) if e['keep_alive']==-1];self.assertEqual(len(seen),2)
            self.assertEqual([e['input'] for e in seen],[[],[]])
            self.assertEqual([e['path'] for e in seen],['/api/embed','/api/embed'])
            self.assertEqual(executions(case),[])
            self.assertEqual({l['model'] for l in p.managed_lanes(case.status())},{p.MODEL,p.EMBED_MODEL})
            p.control(case.socket_path,{'action':'release','token':result['lease']['token']})
    def test_older_empty_input_no_load_cannot_publish_ready_or_fallback_inference(self):
        # Emulate old native semantics: return empty embeddings before loading.
        # All broker process/residency/cleanup code remains the real candidate.
        with tempfile.TemporaryDirectory() as directory:
            wrapper=pathlib.Path(directory)/'empty-return-before-load.py'
            wrapper.write_text("""#!/usr/bin/env python3
import importlib.machinery,importlib.util,sys
loader=importlib.machinery.SourceFileLoader('older_empty_native',%r)
spec=importlib.util.spec_from_loader(loader.name,loader)
n=importlib.util.module_from_spec(spec);sys.modules[loader.name]=n;loader.exec_module(n)
original=n.backend_json_at
def metadata(*args,**kwargs):
    if args[2:4]==('POST','/api/embed') and args[4].get('input')==[]:
        return {'embeddings':[]}
    return original(*args,**kwargs)
n.backend_json_at=metadata
raise SystemExit(n.main())
""" % HELPER)
            wrapper.chmod(0o755)
            with p.PoolHarness(str(wrapper),FIXTURE_BIN,max_servers=2,tags=[EMBED],selected_gpus=GPUS,max_context=8192) as case:
                status,response,_=case.capacity(p.EMBED_MODEL,endpoint='/api/embed',gpu_uuids=[GPUS[0]])
                self.assertEqual(status,503,response)
                self.assertIn('resident',response['error'])
                self.assertEqual(p.managed_lanes(case.status()),[])
                self.assertEqual(executions(case),[])
                self.assertEqual(requests(case),[])
                self.assertEqual([e['kind'] for e in p.events(case.event_log) if e['kind'] in ('start','stop')],['start','stop'])
    def test_original_pool_embedding_inference_regression(self):
        p.test_embedding_model_uses_embedding_warmup(HELPER,FIXTURE_BIN)


if __name__=='__main__':unittest.main()
