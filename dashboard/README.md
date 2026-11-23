# PCB Lab local dashboard

Local browser UI for an ESP32-S3 connected over USB. The layout follows the approved [Superdesign draft](https://p.superdesign.dev/draft/cbbbd115-c956-4b83-8bac-07db1141bb31). Runtime HTML, CSS and JavaScript are local: no Node build, external CDN, cloud service, or TensorFlow installation is required.

## Setup

Python 3.10 or later. From the repository root on Windows:

```powershell
py -3 -m venv dashboard/.venv
dashboard/.venv/Scripts/python.exe -m pip install -r dashboard/requirements.txt
./dashboard/start.ps1
```

Or run `dashboard/.venv/Scripts/python.exe dashboard/server.py` directly. On macOS/Linux:

```sh
python3 -m venv dashboard/.venv
dashboard/.venv/bin/python -m pip install -r dashboard/requirements.txt
dashboard/.venv/bin/python dashboard/server.py
```

Open **http://127.0.0.1:8765**. The service binds only to loopback. Stop it with Ctrl+C. Close other serial monitors before connecting. Opening a serial port can reset some boards; this app never flashes firmware.

## Inspect

1. Plug in the device, refresh ports, select its port, and connect (115200 baud).
2. Watch the serial log for firmware identification. If boot messages were missed, press the physical reset button while connected.
3. With the older USB center-crop firmware identified, select **Start preview**. Save frame downloads the raw PNG and its JSON metadata; browsers may ask to allow multiple downloads.
4. Reference/XOR tabs explain that their image streams are unavailable. The app does not invent boxes or run a second model on the host.

## Supported protocols

| Firmware | Support |
| --- | --- |
| USB camera prototype | Auto-identifies `Camera ready: center...`; sends `V` only after Start preview; reads `FRAME width height bytes score inference_us` followed by exactly width × height grayscale bytes. Displays image and score. |
| Text inference telemetry | Records `inference_us=N`, `free_heap=N`, `min_heap=N`; excludes `stored_test` records. |
| XOR cascade benchmark | Identifies `PCB defect cascade benchmark`; preserves raw logs and parses internal/PSRAM free-memory lines. Its timing format and tile/image results still need a dedicated adapter; it is **not** yet a full cascade benchmark integration. |

Unknown firmware stays passive. No commands are sent until a supported camera protocol has been identified and preview explicitly enabled. Firmware identifiers and parser logic live in `server.py`.

## Benchmark workflow

Choose a configuration label (for example, model revision + gate off), warm-up count, and measured sample count. Start becomes available after compatible timing telemetry arrives. Keep the inspected input and capture conditions controlled; the app collects samples but does not upload a fixed dataset to the board. Preview frames use the inference duration reported in each frame header; accompanying text inference lines are excluded while preview is active to avoid double counting.

Export **each run** before starting another: samples are in memory, the next run replaces them, and stopping the server clears them. JSON includes run metadata, samples, current state, and the bounded evidence log; CSV includes per-sample timing, configuration, port, firmware and available memory readings. Save log also works without benchmark samples. Interrupted runs retain partial samples and are labeled accordingly.

Metrics:
- Median/p95 use measured samples only, excluding warm-up; p95 uses linear interpolation.
- Inference capacity = 1000 / mean inference milliseconds. It is not observed board throughput.
- USB request-to-image time includes device wait/capture, transfer and host PNG encoding; it is not device inference latency or browser display latency.
- Free/minimum free heap and PSRAM come from device logs. They do not isolate model memory, CPU utilization, or peak activation memory.
- No energy, accuracy degradation, or per-class recall is computed. Those need a power measurement and/or matched labeled host/device outputs.

For the course ablations, use the same input set and model with gate off/on, then raw XOR/3×3 erosion. Configuration labels document these externally selected firmware settings; they do not change firmware. A full fixed-dataset driver, cascade result parser, saved multi-run comparisons, accuracy evaluation and power-meter integration remain follow-up work.

## Development and tests

```powershell
dashboard/.venv/Scripts/python.exe -m unittest discover -s dashboard -p test_server.py -v
node --check dashboard/app.js
```

Tests use a fake serial transport to verify warm-up exclusion, percentiles, exact sample counts, malformed frames, binary frame decoding, duplicate suppression and disconnect handling. They do not replace physical hardware validation.

- `server.py`: standard-library HTTP server and one serial reader; `Device` owns connection/run state.
- `index.html`, `style.css`, `app.js`: offline frontend, responsive layout, polling and downloads.
- `superdesign-draft.html`: original design reference only; it uses remote assets and is not served.
- `test_server.py`: hardware-independent protocol tests.

The API is local and single-user. `GET /api/ports`, `/api/state`, `/api/export`, `/api/export.csv`; JSON `POST /api/connect`, `/api/disconnect`, `/api/camera`, `/api/start`, `/api/stop`. Multiple browser tabs share one connection and run. Keep production adaptations in this folder so it can be copied into another project without the legacy viewer or training dependencies.
