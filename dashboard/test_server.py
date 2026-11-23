"""Protocol and benchmark tests; no physical device required."""
import io
import http.client
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import server

class FakeSerial:
    name = 'TEST'
    is_open = True
    def __init__(self, data=b''):
        self.data=io.BytesIO(data)
        self.writes=[]
    def read_until(self, ending, size):
        return self.data.readline(size)
    def read(self, n): return self.data.read(n)
    def write(self, data): self.writes.append(data)
    def close(self): self.is_open=False

class DeviceTests(unittest.TestCase):
    def test_warmup_percentiles_and_exact_sample_limit(self):
        d=server.Device(); d.port=FakeSerial()
        d.start({'count':3,'warmup':1,'configuration':'gate on'})
        for n in (999,10,20,30,500): d.record({'inference_ms':n})
        s=d.state()
        self.assertEqual(s['count'],3)
        self.assertEqual(s['median_ms'],20)
        self.assertEqual(s['p95_ms'],29)
        self.assertEqual(s['inference_per_s'],50)
        self.assertEqual(s['run']['status'],'complete')
        self.assertEqual(d.samples[0]['configuration'],'gate on')

    def test_invalid_sample_counts(self):
        d=server.Device(); d.port=FakeSerial()
        for value in (0,10001,1.5,True):
            with self.assertRaises(ValueError): d.start({'count':value})

    def test_duplicate_start_preserves_run(self):
        d=server.Device(); d.port=FakeSerial(); d.start({'count':2})
        with self.assertRaises(ValueError): d.start({'count':3})
        self.assertEqual(d.run['target'],2)

    def test_camera_binary_frame_and_no_double_count(self):
        d=server.Device()
        d.port=FakeSerial(b'camera defect=0.4 inference_us=12000 free_heap=4096 min_heap=2048\nFRAME 2 2 4 0.4 12000\n'+bytes([0,10,255,80])+b'\n')
        d.camera=True; d.start({'count':1,'warmup':0})
        original=d.record
        def record(sample): original(sample); d.stop.set()
        d.record=record; d.read()
        self.assertEqual(len(d.samples),1)
        self.assertEqual(d.samples[0]['inference_ms'],12)
        self.assertEqual(d.samples[0]['free_heap'],4096)
        self.assertEqual(d.frame['width'],2)
        self.assertTrue(d.frame['image'].startswith('data:image/png;base64,'))
        self.assertEqual(d.port.writes,[b'V'])

    def test_invalid_binary_header_disconnects(self):
        d=server.Device(); d.port=FakeSerial(b'FRAME 9999 1 9999 0.4 100\n')
        d.read()
        self.assertIn('Invalid camera',d.error)
        self.assertFalse(d.port.is_open)

    def test_stored_input_is_not_live_sample(self):
        d=server.Device(); d.port=FakeSerial(b'stored_test defect=0.5 inference_us=100\ncamera defect=0.5 inference_us=200 free_heap=4096\n')
        d.start({'count':1,'warmup':0})
        original=d.record
        def record(sample): original(sample); d.stop.set()
        d.record=record; d.read()
        self.assertEqual(d.samples[0]['inference_ms'],.2)
        self.assertEqual(d.samples[0]['source'],'serial_inference')

    def test_disconnect_interrupts_run(self):
        d=server.Device(); d.port=FakeSerial(); d.start({'count':2})
        d.disconnect()
        self.assertEqual(d.run['status'],'interrupted')
        self.assertFalse(d.state()['connected'])

class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http=server.LocalServer(('127.0.0.1',0),server.Handler)
        cls.worker=threading.Thread(target=cls.http.serve_forever,daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close(); cls.worker.join()

    def request(self, method, path, body=None, extra=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.http.server_port,timeout=3)
        headers={'Host':'127.0.0.1:8765','Content-Type':'application/json'}
        headers.update(extra or {})
        connection.request(method,path,json.dumps(body) if body is not None else None,headers)
        response=connection.getresponse(); result=(response.status,response.read())
        connection.close(); return result

    def test_static_and_disconnected_state(self):
        status,body=self.request('GET','/')
        self.assertEqual(status,200); self.assertIn(b'Inspection workspace',body)
        status,body=self.request('GET','/api/state')
        self.assertEqual(status,200); self.assertFalse(json.loads(body)['connected'])

    def test_json_csv_export(self):
        with patch.object(server.device,'samples',[{'inference_ms':12,'configuration':'gate on'}]):
            status,body=self.request('GET','/api/export')
            self.assertEqual(status,200)
            self.assertEqual(json.loads(body)['samples'][0]['inference_ms'],12)
            status,body=self.request('GET','/api/export.csv')
            self.assertEqual(status,200); self.assertIn(b'12,gate on',body)

    def test_reject_cross_origin_and_disconnected_run(self):
        status,_=self.request('POST','/api/start',{}, {'Origin':'https://example.com'})
        self.assertEqual(status,403)
        status,body=self.request('POST','/api/start',{})
        self.assertEqual(status,400); self.assertIn(b'Connect the device first',body)

if __name__=='__main__': unittest.main()
