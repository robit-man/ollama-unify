import importlib.machinery, importlib.util, json, sys, threading, unittest, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TARGET=Path(sys.argv.pop(1))
loader=importlib.machinery.SourceFileLoader('ttl_broker_under_test',str(TARGET))
spec=importlib.util.spec_from_loader(loader.name,loader)
m=importlib.util.module_from_spec(spec)
sys.modules[loader.name]=m
loader.exec_module(m)
m.REQUEST_ACTIVITY_TTL=10.0
m.REQUEST_DETACHED_TTL=30.0
m.REQUEST_CANCEL_GRACE=5.0

class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.b=m.Broker.__new__(m.Broker)
        self.b.cv=threading.Condition()
        self.b.stopping=threading.Event()
        self.b.active_request_records={}
        self.b.request_activity_renewed_total=0
        self.b.request_disconnected_total=0
        self.b.request_cancelled_total=0
        self.b.request_expired_total=0
        self.b.logical_tombstones={}
        self.tombstones=[]
        self.b._record_logical_tombstone_locked=lambda *x:self.tombstones.append(x)
        self.cancelled=[]
        self.b._cancel_backend_transport=lambda active:self.cancelled.append(active.request_id)
    def request(self, request_id='own-case', logical='logical-case', detached=None):
        lane=SimpleNamespace(kind='managed',retiring=False,lane_id=request_id+'-lane')
        active=m.ActiveRequest(request_id,lane,logical,80.0,95.0,105.0,detached_at=detached)
        self.b.active_request_records[request_id]=active
        admission=m.Admission(lane,request_id,logical,'fingerprint',0,0,1)
        return active,admission
    def at(self,now): return patch.object(m.time,'monotonic',return_value=float(now))
    def test_connected_chunks_keep_activity_lease_renewable(self):
        active,a=self.request()
        for now in (100,125,150):
            with self.at(now): self.assertTrue(self.b.renew_request_activity(a,'response_body'))
            self.assertEqual(active.expires_at,now+10)
            self.assertEqual(active.last_activity_at,now)
        self.assertEqual(self.b.request_activity_renewed_total,3)
    def test_connected_backend_start_renews_activity_lease(self):
        active,a=self.request(); backend=object()
        with self.at(125): self.assertTrue(self.b.request_backend_started(a,backend))
        self.assertIs(active.backend,backend)
        self.assertTrue(active.backend_started)
        self.assertEqual(active.expires_at,135)
    def test_detached_chunks_cannot_move_disconnect_deadline(self):
        active,a=self.request()
        with self.at(100): self.b.request_client_detached(a)
        self.assertEqual(active.expires_at,130)
        for now in (105,129,131,1000):
            with self.at(now): self.assertTrue(self.b.renew_request_activity(a,'response_body'))
            self.assertEqual(active.expires_at,130)
            self.assertEqual(active.last_activity_at,now)
        self.assertEqual(self.b.request_activity_renewed_total,4)
    def test_detached_backend_start_cannot_move_disconnect_deadline(self):
        active,a=self.request()
        with self.at(100): self.b.request_client_detached(a)
        with self.at(129): self.assertTrue(self.b.request_backend_started(a,object()))
        self.assertEqual(active.expires_at,130)
    def test_monotonic_zero_disconnect_is_a_real_disconnect(self):
        active,a=self.request(detached=0.0)
        with self.at(29): self.b.renew_request_activity(a,'response_body')
        self.assertEqual(active.expires_at,30)
    def test_duplicate_detach_does_not_restart_window(self):
        active,a=self.request()
        with self.at(100): self.b.request_client_detached(a)
        with self.at(120): self.b.request_client_detached(a)
        self.assertEqual(active.detached_at,100)
        self.assertEqual(active.expires_at,130)
        self.assertEqual(self.b.request_disconnected_total,1)
    def test_nonlogical_disconnect_still_cancels_immediately(self):
        active,a=self.request(logical='')
        with self.at(100): self.b.request_client_detached(a)
        self.assertEqual(self.cancelled,['own-case'])
        self.assertEqual(active.cancel_reason,'client_disconnected')
        self.assertEqual(active.cancel_requested_at,100)
        self.assertEqual(active.expires_at,105)
    def test_missing_or_cancelled_records_cannot_be_renewed(self):
        active,a=self.request(); active.cancel_requested_at=99
        with self.at(100):
            self.assertFalse(self.b.renew_request_activity(a,'response_body'))
            self.assertFalse(self.b.request_backend_started(a,object()))
        self.assertEqual(active.expires_at,105)
        self.assertEqual(self.b.request_activity_renewed_total,0)
        self.b.active_request_records.clear()
        with self.at(100):
            self.assertFalse(self.b.renew_request_activity(a,'response_body'))
            self.assertFalse(self.b.request_backend_started(a,object()))
    def test_watchdog_expires_only_exact_detached_owner(self):
        active,a=self.request(detached=100.0); active.expires_at=130
        foreign,f=self.request('foreign-work','foreign-logical')
        foreign.expires_at=145
        def cancel(request):
            self.cancelled.append(request.request_id)
            self.b.stopping.set()
        self.b._cancel_backend_transport=cancel
        with self.at(131): self.b.active_request_watchdog()
        self.assertEqual(self.cancelled,['own-case'])
        self.assertEqual(active.cancel_reason,'detached_request_expired')
        self.assertEqual(self.b.request_expired_total,1)
        self.assertEqual(self.tombstones,[('logical-case','own-case','detached_request_expired')])
        self.assertIsNone(foreign.cancel_requested_at)
        self.assertEqual(foreign.expires_at,145)
        self.assertFalse(foreign.lane.retiring)
    def test_connected_foreign_owner_renews_independently(self):
        active,a=self.request(detached=100.0);active.expires_at=130
        foreign,f=self.request('foreign-work','foreign-logical')
        with self.at(129): self.b.renew_request_activity(f,'response_body')
        self.assertEqual(foreign.expires_at,139)
        self.assertEqual(active.expires_at,130)
    def test_successful_completion_preserves_terminal_finalization_grace(self):
        active,a=self.request(detached=100.0); active.expires_at=130
        with self.at(120): self.assertTrue(self.b.request_backend_complete(a))
        self.assertTrue(active.backend_completed)
        self.assertEqual(active.expires_at,125)
        with self.at(121): self.b.request_client_detached(a)
        self.assertEqual(active.expires_at,125)

    def test_fresh_heartbeat_restores_external_lease_even_past_expected_release(self):
        payload={'leases':[{'token':'private-fixture-token','owner':'foreign-external',
            'state':'active','requested_mib':61440,'created_at':10.0,
            'heartbeat_at':990.0,'ttl':300,'gpu_uuids':['GPU-fixture-1'],
            'justification':'independent external service','expected_release_at':100.0}]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'leases.json'; path.write_text(json.dumps(payload))
            with patch.object(m,'LEASE_STATE_PATH',path),patch.object(m.time,'time',return_value=1000.0):
                restored=self.b._load_leases()
            self.assertEqual(path.read_text(),json.dumps(payload))
        lease=restored['private-fixture-token']
        self.assertEqual(lease.state,'active')
        self.assertEqual(lease.owner,'foreign-external')
        self.assertEqual(lease.gpu_uuids,['GPU-fixture-1'])
        self.assertEqual(lease.heartbeat_at,990.0)
        self.assertEqual(lease.expected_release_at,100.0)
    def test_scoped_stop_targets_only_named_managed_lane(self):
        target=SimpleNamespace(kind='managed',lane_id='own-lane',in_flight=1,retiring=False,process=object())
        foreign=SimpleNamespace(kind='managed',lane_id='foreign-lane',in_flight=0,retiring=False)
        self.b.lanes={'own-lane':target,'foreign-lane':foreign}
        self.b._prune_dead_lanes_locked=lambda:None
        self.b.transition=threading.RLock()
        with patch.object(m,'gpu_snapshot',return_value=[]), \
                patch.object(m,'process_group_alive',return_value=False) as alive:
            result=self.b.stop_lane('own-lane',force=True)
        self.assertEqual(result['stopped_lanes'],['own-lane'])
        alive.assert_called_once_with(target.process)
        self.assertEqual(self.b.lanes,{'foreign-lane':foreign})
        self.assertFalse(foreign.retiring)
    def test_shutdown_preserves_external_lease_table_but_stops_managed_pool(self):
        self.b.leases={'external-fixture':object()}
        leases=self.b.leases.copy()
        calls=[]
        self.b.stop_pool_lanes=lambda reason:calls.append(reason)
        self.b.shutdown()
        self.assertTrue(self.b.stopping.is_set())
        self.assertEqual(calls,['broker shutdown'])
        self.assertEqual(self.b.leases,leases)

if __name__=='__main__':
    unittest.main(verbosity=2)
