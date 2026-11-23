"""Local-only USB inspection dashboard. Run with the project's Python environment."""
import base64
import csv
import io
import json
import math
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
import serial
from serial.tools import list_ports
from PIL import Image

ROOT = Path(__file__).resolve().parent

def percentile(values, q):
    a = sorted(values)
    if not a:
        return None
    p = (len(a)-1)*q
    return a[math.floor(p)] + (a[math.ceil(p)]-a[math.floor(p)])*(p % 1)

class Device:
    def __init__(self):
        self.lock = threading.RLock()
        self.port = None
        self.thread = None
        self.stop = threading.Event()
        self.logs = deque(maxlen=160)
        self.samples = []
        self.frame = None
        self.error = None
        self.firmware = 'Unknown'
        self.memory = {}
        self.run = None
        self.camera = False
        self.requested = None
        self.latest = None

    def disconnect(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Serial reader is still stopping; retry shortly.')
        if self.port:
            self.port.close()
        self.port = None
        self.camera = False
        self.requested = None
        if self.run and self.run['status'] == 'running':
            self.run['status'] = 'interrupted'

    def connect(self, name):
        if name not in [p.device for p in list_ports.comports()]:
            raise ValueError('Select an available serial port.')
        self.disconnect()
        self.port = serial.Serial(name, 115200, timeout=.3, write_timeout=1)
        self.stop.clear()
        self.error = None
        self.firmware = 'Unknown'
        self.frame = None
        self.latest = None
        self.memory = {}
        self.camera = False
        self.requested = None
        self.logs.clear()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read_exact(self, n):
        data = bytearray()
        deadline = time.monotonic()+12
        while len(data)<n and not self.stop.is_set() and time.monotonic()<deadline:
            data.extend(self.port.read(n-len(data)))
        if len(data)!=n:
            raise TimeoutError('Incomplete camera frame; reconnect the device.')
        return bytes(data)

    def record(self, sample):
        with self.lock:
            self.latest = sample
            if not self.run or self.run['status'] != 'running':
                return
            if self.run['warmup_left']:
                self.run['warmup_left'] -= 1
                return
            sample = dict(sample, **self.memory, configuration=self.run['configuration'],
                          firmware=self.firmware, port=self.port.name,
                          run_id=self.run['id'], sample=len(self.samples)+1)
            self.samples.append(sample)
            if len(self.samples) >= self.run['target']:
                self.run['status'] = 'complete'
                self.run['duration_s'] = time.monotonic()-self.run['_start']

    def read(self):
        pending = bytearray()
        try:
            while not self.stop.is_set():
                if self.camera and self.requested is None:
                    self.requested = time.monotonic()
                    self.port.write(b'V')
                part = self.port.read_until(b'\n', 4096)
                pending.extend(part)
                if len(pending)>8192:
                    raise ValueError('Unrecognized serial stream. Reconnect with compatible firmware.')
                if not pending.endswith(b'\n'):
                    if self.requested and time.monotonic()-self.requested>15:
                        raise TimeoutError('No frame received after camera request.')
                    continue
                raw = bytes(pending); pending.clear()
                line = raw.decode('utf-8', errors='replace').strip()
                if line.startswith('FRAME '):
                    _, w, h, n, score, us = line.split()
                    w,h,n = int(w),int(h),int(n)
                    score,us = float(score),int(us)
                    if not (0<w<=640 and 0<h<=480 and n==w*h and math.isfinite(score) and us>=0):
                        raise ValueError('Invalid camera frame header.')
                    pixels = self.read_exact(n)
                    image = Image.frombytes('L', (w,h), pixels)
                    buf = io.BytesIO(); image.save(buf, format='PNG')
                    sample = {'timestamp':datetime.now(timezone.utc).isoformat(), 'source':'camera_frame',
                              'inference_ms':us/1000, 'score':score,
                              'roundtrip_ms':(time.monotonic()-self.requested)*1000 if self.requested else None}
                    with self.lock:
                        self.frame = {'image':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
                                      'width':w, 'height':h, **sample}
                        self.firmware = 'USB center-crop prototype'
                    if self.camera or self.requested is not None:
                        self.record(sample)
                    self.requested = None
                    continue
                if not line:
                    continue
                with self.lock:
                    self.logs.append(line)
                    for key in ('free_heap', 'min_heap'):
                        match = re.search(r'\b'+key+r'=(\d+)', line)
                        if match: self.memory[key] = int(match[1])
                    mem = re.search(r'internal free\s+(\d+) \(min\s+(\d+)\)\s+psram free\s+(\d+) \(min\s+(\d+)\)',line)
                    if mem:
                        self.memory.update(dict(zip(('free_heap','min_heap','free_psram','min_psram'),map(int,mem.groups()))))
                    if 'PCB defect cascade benchmark' in line:
                        self.firmware = 'XOR cascade benchmark (logs only)'
                    elif 'Camera ready: center' in line:
                        self.firmware = 'USB center-crop prototype'
                match = re.search(r'\binference_us=(\d+)', line)
                if match and not self.camera and self.requested is None and 'stored_test' not in line:
                    self.record({'timestamp':datetime.now(timezone.utc).isoformat(),
                                 'source':'serial_inference', 'inference_ms':int(match[1])/1000,
                                 'roundtrip_ms':None})
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                if self.run and self.run['status']=='running': self.run['status']='interrupted'
            self.port.close()

    def start(self, body):
        if not self.port or not self.port.is_open:
            raise ValueError('Connect the device first.')
        if self.run and self.run['status']=='running':
            raise ValueError('Stop the current run first.')
        count, warmup = body.get('count',30), body.get('warmup',5)
        if type(count) is not int or type(warmup) is not int:
            raise ValueError('Sample counts must be whole numbers.')
        if not 1<=count<=10000 or not 0<=warmup<=1000:
            raise ValueError('Samples must be 1–10000; warm-up 0–1000.')
        self.samples = []
        self.run = {'id':datetime.now(timezone.utc).isoformat(), 'status':'running',
                    'target':count, 'warmup_left':warmup, 'warmup':warmup,
                    'configuration':str(body.get('configuration','Current firmware'))[:120],
                    '_start':time.monotonic()}

    def state(self):
        with self.lock:
            values = [s['inference_ms'] for s in self.samples]
            return {'connected':bool(self.port and self.port.is_open), 'firmware':self.firmware,
                    'error':self.error, 'camera':self.camera, 'frame':self.frame, 'latest':self.latest,
                    'memory':dict(self.memory), 'logs':list(self.logs),
                    'run':{k:v for k,v in self.run.items() if not k.startswith('_')} if self.run else None,
                    'count':len(values), 'median_ms':percentile(values,.5), 'p95_ms':percentile(values,.95),
                    'inference_per_s':1000/(sum(values)/len(values)) if values and sum(values)>0 else None}

device = Device()
commands = threading.Lock()

class LocalServer(ThreadingHTTPServer):
    def server_bind(self):
        # Avoid reverse DNS lookup during startup on offline lab machines.
        TCPServer.server_bind(self)
        self.server_name = 'localhost'
        self.server_port = self.server_address[1]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def send(self, status, data, kind='application/json'):
        if not isinstance(data,bytes): data=json.dumps(data,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.headers.get('Host') not in ('127.0.0.1:8765','localhost:8765'):
            return self.send(403,{'error':'Host rejected'})
        path=self.path.split('?')[0]
        if path=='/api/state': return self.send(200,device.state())
        if path=='/api/ports': return self.send(200,[{'port':p.device,'description':p.description} for p in list_ports.comports()])
        if path=='/api/export':
            with device.lock:
                return self.send(200,{'schema_version':1,'state':device.state(),'samples':list(device.samples),
                                     'scope':'Measured device inference only. No ground truth, energy measurement, or full-board accuracy is inferred.'})
        if path=='/api/export.csv':
            with device.lock: rows=list(device.samples)
            output=io.StringIO()
            if rows:
                writer=csv.DictWriter(output,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader(); writer.writerows(rows)
            return self.send(200,output.getvalue().encode(),'text/csv; charset=utf-8')
        files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if path in files:
            kind={'/':'text/html','/app.js':'text/javascript','/style.css':'text/css'}[path]
            return self.send(200,(ROOT/files[path]).read_bytes(),kind+'; charset=utf-8')
        self.send(404,{'error':'Not found'})

    def do_POST(self):
        if self.headers.get('Host') not in ('127.0.0.1:8765','localhost:8765'):
            return self.send(403,{'error':'Host rejected'})
        # Browser mutations must originate from this loopback server.
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8765','http://localhost:8765'):
            return self.send(403,{'error':'Origin rejected'})
        if self.headers.get('Content-Type') != 'application/json':
            return self.send(415,{'error':'JSON required'})
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=4096: raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(size))
            with commands:
                if self.path=='/api/connect': device.connect(body['port'])
                elif self.path=='/api/disconnect': device.disconnect()
                elif self.path=='/api/camera':
                    if not device.port or not device.port.is_open: raise ValueError('Connect the device first.')
                    if device.firmware!='USB center-crop prototype':
                        raise ValueError('Camera protocol is not identified. Wait for firmware logs or reconnect after resetting the board.')
                    if device.run and device.run['status']=='running': raise ValueError('Stop the run before changing capture mode.')
                    device.camera=bool(body.get('enabled'))
                elif self.path=='/api/start':
                    with device.lock: device.start(body)
                elif self.path=='/api/stop':
                    with device.lock:
                        if device.run and device.run['status']=='running': device.run['status']='stopped'
                else: return self.send(404,{'error':'Not found'})
            self.send(200,{'ok':True})
        except Exception as exc: self.send(400,{'error':str(exc)})

if __name__=='__main__':
    server=LocalServer(('127.0.0.1',8765),Handler)
    print('PCB Lab dashboard: http://127.0.0.1:8765',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close(); device.disconnect()
