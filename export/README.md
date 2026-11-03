# export/

Quantization and conversion to the on-chip runtime.

**Toolchain decision (checkpoint 2, 2026-11-03):** ESP-DL, not TFLite Micro.

```
PyTorch -> ONNX (opset 18) -> ESP-PPQ -> .espdl -> ESP-DL -> ESP-IDF
```

The fallback in the plan (`PyTorch -> TFLite -> TFLM + ESP-NN`) is **not needed**: every
operator in the student is supported by ESP-DL in int8, and ESP-DL additionally provides the
PC-vs-device parity check and the per-layer latency/memory profiling the plan requires as
built-ins. See the project README for the evidence.

| File | Purpose |
| --- | --- |
| `operators.py` | ESP-DL's int8 operator list, vendored with provenance, plus the `groups` restriction check |
| `to_espdl.py` | ONNX export (+ artefact stripping, operator check) and the ESP-PPQ stage |

## Usage

ONNX only — runs on any interpreter, and is what CI checks:

```bash
python export/to_espdl.py --in-channels 1 --out export/artifacts
```

Full path to a deployable model (needs esp-ppq, `python >=3.11,<3.13`):

```bash
python export/to_espdl.py --in-channels 1 --espdl --weights runs/control_s0/best.pt --npz data/tiles/deeppcb_tiles_t160_s120.npz --out firmware/pcb_bench/models --name student_c1
```

Three files land beside each other: the `.espdl` binary, a `.info` text dump (structure,
quantized weights, embedded test input/output) and a `.json` of quantization parameters.

## Things that bite

* **ESP32-S3 quantizes per-tensor, symmetric, with power-of-two scales.** The P4 gets
  per-channel Conv/Gemm; the S3 does not. Per-tensor weight scales are the known weak point
  of depthwise convolutions, so expect the depthwise layers to carry most of the int8 error.
  Read `error_report` before blaming the architecture. If w8a8 loses too much, the documented
  escalation is `w8a16` or per-layer promotion via `QuantizationSetting.dispatching_table` —
  but whichever is chosen must then be used for **every** run in the comparison.
* **The pooled binary input quantizes losslessly.** Values are `{0, ¼, ½, ¾, 1}`, all exactly
  representable at `exponent = -6`.
* **An untrained model exports `Identity` nodes, which ESP-DL does not implement.** With
  default BatchNorm statistics every folded convolution bias is the same zero vector, ONNX
  deduplicates them, and the duplicates become Identity aliases. `strip_identity()` removes
  them; a trained model never produces them. This matters because the plan calls for pushing
  an *untrained* student through the path first.
* **The dynamo exporter writes an `.onnx.data` sidecar by default.** Both exporters are
  forced to emit a single self-contained file, or copying the `.onnx` to Colab would quietly
  lose the weights.
* **BatchNorm must fold away** — ESP-DL has no `BatchNormalization` kernel. The student's
  convolutions carry no bias precisely so that folding is clean.
* **`batch_size` is 1 on device.** Only PC-side evaluation can batch.
* Calibration tiles are drawn from the **training fold only**, so quantization never sees
  validation or test data.
