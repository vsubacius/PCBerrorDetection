'use strict';
const $ = id => document.getElementById(id);
let state = null, view = 'camera', busy = false;
const fmt = (n, unit = '', places = 1) => n == null ? '—' : `${Number(n).toFixed(places)}${unit}`;
function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
async function api(path, body) {
  const response = await fetch(`/api/${path}`, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
async function action(fn) {
  if (busy) return;
  busy = true; error();
  try { await fn(); await pollOnce(); } catch (e) { error(e.message); }
  finally { busy = false; render(); }
}
function download(content, name, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function ports() {
  const previous = $('port').value, items = await api('ports');
  $('port').replaceChildren(new Option(items.length ? 'Select port…' : 'No serial ports found', ''));
  for (const p of items) $('port').add(new Option(`${p.port} · ${p.description}`, p.port));
  if (items.some(p => p.port === previous)) $('port').value = previous;
  else if (items.length === 1) $('port').value = items[0].port;
}
function render() {
  if (!state) return;
  const s = state, running = s.run?.status === 'running';
  $('connection').textContent = s.connected ? 'Connected' : 'Disconnected';
  $('connection').classList.toggle('online', s.connected);
  $('connect').textContent = s.connected ? 'Disconnect' : 'Connect device';
  $('connect').disabled = busy; $('port').disabled = s.connected || busy;
  $('firmware').textContent = s.firmware;
  $('start').disabled = !s.connected || !s.latest || running || busy;
  $('stop').hidden = !running;
  for (const id of ['configuration','count','warmup']) $(id).disabled = running;
  $('camera').disabled = !s.connected || s.firmware !== 'USB center-crop prototype' || running || busy;
  $('camera').textContent = s.camera ? 'Pause preview' : 'Start preview';
  $('snapshot').disabled = !s.frame;
  $('csv').disabled = !s.count;
  $('median').textContent = fmt(s.median_ms,' ms'); $('p95').textContent = fmt(s.p95_ms,' ms');
  $('capacity').textContent = fmt(s.inference_per_s,' /s');
  $('latest').textContent = fmt(s.latest?.inference_ms,' ms');
  $('roundtrip').textContent = fmt(s.frame?.roundtrip_ms,' ms');
  $('score').textContent = fmt(s.frame?.score,'',3);
  for (const [id,key] of [['heap','free_heap'],['min-heap','min_heap'],['psram','free_psram']]) $(id).textContent = fmt(s.memory[key] == null ? null : s.memory[key]/1024,' KiB');
  $('progress').max = s.run?.target || 30; $('progress').value = s.count;
  $('run-status').textContent = s.run ? `${s.run.status} · ${s.count}/${s.run.target} samples${s.run.warmup_left && running ? ` · ${s.run.warmup_left} warm-up remaining` : ''}` : s.connected ? (s.latest ? 'Ready to record.' : 'Waiting for compatible inference telemetry…') : 'Connect a device to begin.';
  const logText = s.logs.join('\n') || (s.connected ? 'Listening at 115200 baud…' : 'Waiting for connection…');
  if ($('log').textContent !== logText) { const nearEnd = $('log').scrollHeight-$('log').scrollTop-$('log').clientHeight<40; $('log').textContent = logText; if (nearEnd) $('log').scrollTop=$('log').scrollHeight; }
  const hasFrame = view === 'camera' && Boolean(s.frame);
  $('frame').hidden = !hasFrame; $('empty').hidden = hasFrame;
  if (hasFrame && $('frame').getAttribute('src') !== s.frame.image) $('frame').src = s.frame.image;
  if (view !== 'camera') {
    $('empty-title').textContent = `${view === 'xor' ? 'XOR' : 'Reference'} image unavailable`;
    $('empty-copy').textContent = 'The current device adapter does not transmit this image. Add a firmware protocol adapter to enable this view.';
  } else {
    $('empty-title').textContent = s.connected ? 'Waiting for camera image' : 'Device disconnected';
    $('empty-copy').textContent = s.connected ? (s.firmware === 'USB center-crop prototype' ? 'Select Start preview to request camera frames.' : 'Listening for firmware identification. If startup logs were missed, press the board reset button while connected.') : 'Connect an ESP32-S3 via USB and select its port to begin inspection.';
  }
  $('frame-info').textContent = s.frame ? `${s.frame.width} × ${s.frame.height} · Grayscale · ${s.camera && s.connected ? 'Live' : 'Last captured frame'}` : 'Resolution: — · Format: —';
}
async function pollOnce() { state = await api('state'); render(); if (state.error) error(state.error); }
async function poll() { try { await pollOnce(); } catch (e) { error(`Local service unavailable: ${e.message}`); $('connection').textContent='Service unavailable'; $('connection').classList.remove('online'); for (const id of ['connect','start','camera']) $(id).disabled=true; } finally { setTimeout(poll,600); } }
$('refresh').onclick = () => action(ports);
$('connect').onclick = () => action(async () => { if (!state?.connected && !$('port').value) throw new Error('Select a serial port first.'); await api(state?.connected ? 'disconnect' : 'connect', {port:$('port').value}); });
$('camera').onclick = () => action(() => api('camera',{enabled:!state.camera}));
$('run-form').onsubmit = e => { e.preventDefault(); action(() => api('start',{count:Number($('count').value),warmup:Number($('warmup').value),configuration:$('configuration').value})); };
$('stop').onclick = () => action(() => api('stop',{}));
$('csv').onclick = () => action(async () => { const r=await fetch('/api/export.csv'); if(!r.ok)throw new Error('Export failed'); download(await r.text(),'pcb-benchmark.csv','text/csv'); });
$('json').onclick = () => action(async () => download(JSON.stringify(await api('export'),null,2),'pcb-benchmark.json','application/json'));
$('save-log').onclick = () => download(state?.logs.join('\n') || '', 'pcb-serial-log.txt','text/plain');
$('snapshot').onclick = () => { if(!state?.frame)return; const a=document.createElement('a'); a.href=state.frame.image; a.download='pcb-frame.png'; a.click(); download(JSON.stringify({...state.frame,image:undefined,firmware:state.firmware,scope:'Center-crop score; defect location unknown'},null,2),'pcb-frame.json','application/json'); };
document.querySelectorAll('[data-view]').forEach((button,index,buttons) => {
  button.id ||= `view-${button.dataset.view}`;
  button.onclick = () => { view=button.dataset.view; buttons.forEach(b=>{b.setAttribute('aria-selected',String(b===button));b.tabIndex=b===button?0:-1;});$('viewport').setAttribute('aria-labelledby',button.id);render(); };
  button.tabIndex=index===0?0:-1;
  button.onkeydown=e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const next=buttons[(index+(e.key==='ArrowRight'?1:buttons.length-1))%buttons.length];next.click();next.focus();}};
});
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>{document.querySelectorAll('nav a').forEach(n=>n.classList.toggle('active',n===a));document.querySelector(a.getAttribute('href')).focus();});
ports().catch(e=>error(e.message)); poll();
