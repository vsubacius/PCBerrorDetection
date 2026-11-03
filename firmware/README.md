# firmware/

ESP-IDF projects for the Seeed XIAO ESP32-S3 Sense.

| Project | State |
| --- | --- |
| [`pcb_bench/`](pcb_bench/) | benchmark mode — written, **not yet run on hardware** |
| live demo mode | deferred to ~Nov 29 (see `CLAUDE.md`) |

## `pcb_bench` — benchmark mode

Every chip number this project reports comes from this app. It:

1. records a memory baseline before allocating anything;
2. loads the int8 `.espdl` student and calls `dl::Model::test()`, which checks device
   output against the PC values ESP-PPQ embedded in the model — **the PC-vs-device
   parity check the experiment plan requires**;
3. calls `dl::Model::profile()` for per-layer latency plus a memory breakdown
   (`fbs_model`, `parameter_copy`, `variable`, `others`) across internal RAM, PSRAM and flash;
4. walks all 25 tiles of a board with the gate **off** and then **on**, timing tile
   access, gate, Stage-2 inference and postprocessing separately;
5. reports peak internal SRAM and PSRAM via `heap_caps_get_minimum_free_size()`.

The gate is a **runtime** switch so that the `reference` and `gate_only` rows of the run
matrix come from one binary and one model file, differing in nothing else.

### Status and honest caveats

* The code compiles cleanly on the host where it can (`gate.cpp`, `board.cpp` under
  `-Wall -Wextra -Werror`) and `gate.cpp` is covered by a bit-exactness test against the
  numpy reference — see [`tests/test_gate_parity.py`](../tests/test_gate_parity.py).
* `main.cpp` has **not** been compiled against ESP-IDF or run on a board. Expect small
  fixes on the first `idf.py build`.
* Boards are **synthetic** until the microSD path lands (`board.hpp`). Latency and memory
  are meaningful on synthetic boards; the printed gate skip rate is **not** — it depends on
  real defect statistics. Treat it as a smoke test.

### Build and flash

Needs ESP-IDF `release/v5.3` or newer (an ESP-DL requirement) and a quantized model.

```bash
python export/to_espdl.py --in-channels 1 --espdl \
    --npz data/tiles/deeppcb_tiles_t160_s120.npz \
    --out firmware/pcb_bench/models --name student_c1
```

```bash
cd firmware/pcb_bench && idf.py set-target esp32s3
```

```bash
idf.py build flash monitor
```

The ESP-PPQ stage pins `python >=3.11,<3.13`, so run it in Colab or a 3.12 virtualenv.
`export/to_espdl.py` without `--espdl` stops after ONNX and runs on any interpreter.

### What to report back

The toolchain checkpoint needs these four, from the serial log:

* whether `dl::Model::test()` passes (PC vs device parity);
* `profile_module()` per-layer latency, and the `ms per board` lines for gate off/on;
* `profile_memory()` totals, plus the `PEAK internal SRAM / PSRAM` line;
* the `.espdl` file size (model flash cost).

### Notes that shaped the code

* **Erode the XOR, radius 1.** Raw XOR counting separates clean from defective tiles at only
  AUC 0.78, because re-binarized JPEG leaves a 1-px fringe along every copper edge. A 3×3
  erosion takes it to 0.95. Do **not** use radius 2: a 5×5 erosion also deletes 15.4% of
  *defect* tiles outright (they score exactly zero), capping tile recall near 85%.
* **The erosion streams.** A *k*×*k* binary erosion is a separable sliding AND with no
  multiplies, and the vertical pass only needs `(2r+1)` rows live, so the gate uses
  **480 bytes** of scratch rather than buffering the 25 KB XOR image.
* **Per-tile erosion is fine.** Eroding each tile independently rather than the whole board
  inflates the clean-tile count on 6.8% of clean tiles by a mean of 0.4 px — negligible, so
  no board-wide XOR buffer is needed.
* **NHWC.** ESP-DL runs Conv and the pooling ops in NHWC, so a 2-channel input must be
  interleaved per pixel, not planar. For 1 channel the layouts coincide.
* **No sigmoid.** The model emits logits; the firmware compares against `log(p/(1-p))`.
  That keeps a quantized sigmoid out of the int8 graph entirely.
