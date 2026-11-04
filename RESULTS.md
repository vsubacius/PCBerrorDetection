# Measured results

Every number here was measured in this repo and can be regenerated with the command
given beside it. Nothing is estimated. Where something is pending or blocked it says so
rather than being left blank.

**Status:** PC-side float32 results are complete for the control and the input ablation.
**No on-device numbers exist yet** — the benchmark firmware is written but has never been
run on hardware, and int8 quantization needs `esp-ppq`, which pins `python >=3.11,<3.13`.

Last updated 2026-11-04.

---

## 1. The tile grid: 160 px tiles, stride 120

```bash
python data/grid_sweep.py --split all
```

A defect is findable only if some tile holds ≥70% of its area, so **reachability is a hard
ceiling on defect-level recall**, set purely by geometry. Measured over all 10,013 boxes,
the 5×5 grid at 160/120 leaves only 5 unreachable, and per split:

| split | boards | defects | unreachable | recall ceiling |
| --- | --- | --- | --- | --- |
| trainval | 1,000 | 6,873 | 5 | 99.93% |
| **test (official)** | 500 | 3,140 | **0** | **100.00%** |

All five unreachable defects are in trainval, **none in the official test split**, so the
reported test ceiling is 100%. Two of the five fall in the seed-0 validation fold. Four of
the five sit within 0.011 of the threshold, so `pos_thresh = 0.66` would make all 10,013
reachable — the 0.70 value is still **pending team confirmation**.

The older full-containment rule from `CLAUDE.md` caps recall at 95.17% (trainval) and
95.89% (test) for purely geometric reasons and is **not used**.

### The official test split is not board-disjoint

DeepPCB's 1,500 images come from 11 groups, but only **7 appear in trainval, and all 7 also
appear in test**; the 4 test-only groups supply 332 of the 500 test images. The official
benchmark is therefore a board-overlapping split, which makes the image-level validation
fold the better proxy for test performance, and `--split-by group` a separate
generalisation probe rather than a stricter version of the same thing. With 7 trainval
groups and `val_frac = 0.2`, `assign_folds` holds out exactly **one** group, so read that
check loosely.

---

## 2. Stage-1 gate: erode the XOR, radius 1

```bash
python eval/gate_analysis.py
```

Measured on 40 validation boards, at a 98% defect-tile recall target:

| gate | threshold | clean tiles skipped | note |
| --- | --- | --- | --- |
| raw XOR count | 128 | 55.3% | the 1-px JPEG fringe along every copper edge swamps small defects |
| **3×3 erosion** | **7** | **83.1%** | the classical gate |
| 5×5 erosion | — | — | **cannot reach 98% at any threshold** (ceiling 91.1%) |

**`Findings.md` is wrong that a 5×5 erosion reaches 98% recall.** Directly measured:
**15.4% of defect tiles score exactly 0 under a 5×5 erosion**, versus **0.0% under 3×3**. A
tile scoring 0 only fires under a negative threshold, which fires on every clean tile too,
so those defects are unrecoverable.

The original `gate_analysis.py` concealed this: it derived `thr@98` from a quantile of the
defect-tile scores and never checked the recall that threshold achieves. For `erode2`,
`skip@98 = 98.4%` and `recall@skip50 = 86.9%` described the **same** threshold of zero. The
script now reports a ceiling and the achieved recall, and `n/a` when a target is
unreachable.

Per-tile erosion (what the firmware can do cheaply) inflates the clean-tile count on 6.8%
of clean tiles by a mean of 0.4 px against whole-board erosion — negligible, so no
board-wide XOR buffer is needed.

---

## 3. Stage-2 student: 61K parameters, 1.33M MACs per tile

```bash
python training/models.py --layers
```

| | mode A (1 ch) | modes B/C (2 ch) | team target |
| --- | --- | --- | --- |
| parameters | 60,966 | 61,110 | ~63K |
| int8 weights | 57.5 KB | 57.6 KB | ~63 KB |
| MACs / tile | 1.33 M | 1.56 M | ~1.5 M |
| MACs / 25-tile board | 33.2 M | 39.0 M | — |
| peak activation | 31.2 KB | 37.5 KB | 16 KB |

Parameters and MACs meet the team targets. **The 16 KB activation target was not reachable**
— a 160×160 int8 tile is 25 KB of input buffer alone — and is not binding against 512 KB of
SRAM, so the design keeps full tile resolution instead.

---

## 4. Input ablation: a reference is worth 11.7× the defect recall

```bash
python training/run_matrix.py --set input_ablation
python eval/compare_runs.py --runs runs/control_s* runs/input_B_s* runs/input_C_s* --fold val
```

Nine runs, three seeds per mode, everything else identical (plain masked BCE, no gate, no
teacher, no class weights). Validation fold, float32, thresholds set per run to a 5%
board-level false-alarm target on the same fold's templates run as defect-free boards.

| input mode | macro AP | defect recall | precision | clean-board FA |
| --- | --- | --- | --- | --- |
| **A** test only | 0.4534 ±0.0297 | 7.35% ±2.22 | 81.8% ±8.8 | 3.7% ±1.2 |
| **B** test + template | 0.7994 ±0.0221 | 81.10% ±8.32 | 59.6% ±5.3 | 3.7% ±1.2 |
| **C** test + XOR | 0.8050 ±0.0333 | 86.27% ±9.30 | 56.6% ±12.7 | 4.3% ±0.6 |

Mean ± sample standard deviation over seeds 0/1/2.

**Two conclusions.**

1. **The reference channel is the single largest effect measured so far.** Defect recall
   goes from 7.35% to 86.27% at the same false-alarm budget, roughly **11.7×**, and the gap
   is more than ten times the seed spread. This is the strongest available justification
   for the project's "reference-guided" framing.
2. **B and C are statistically indistinguishable.** Macro AP differs by 0.006 against a
   seed spread of ±0.02–0.03; recall differs by 5.2 points against ±8.3–9.3. Neither
   difference is a result, so the choice between them must be made on other grounds.

### The mechanism, per class

Change in average precision relative to mode A:

| | open | short | mousebite | spur | copper | pin-hole |
| --- | --- | --- | --- | --- | --- | --- |
| **B** − A | +0.440 | +0.499 | +0.267 | +0.409 | +0.206 | +0.255 |
| **C** − A | +0.422 | +0.522 | +0.252 | +0.448 | +0.210 | +0.255 |

Absolute AP in mode C: copper 0.979, pin-hole 0.915, short 0.837, open 0.794, spur 0.710,
mousebite 0.595.

The split is close to the predicted one. **Short, open and spur gain most** (+0.42 to
+0.52): these are *relational* defects, where deciding whether a trace is broken or bridged
requires knowing the intended topology, which a single binarized image does not carry.
**Copper and pin-hole gain least** (+0.21, +0.26): extra metal and a hole are locally
visible without any reference, and mode A already scored 0.77 and 0.66 on them.

One honest correction to the prediction: **mousebite behaves like an intrinsic defect, not
a relational one** (+0.25, the second-smallest gain), and it remains the hardest class in
every mode. A bite out of a trace edge is partly visible as an irregular edge without a
reference, which is presumably why.

### Seed variance is large enough to matter

The three mode-A runs scored 0.4193 / 0.4741 / 0.4669 macro AP — a 13% spread on an
*identical* configuration. Any future effect smaller than about ±0.03 macro AP or ±8 points
of recall cannot be separated from noise at three seeds. This is the main reason the
gate and distillation results will need care.

### Pending

- **Quantization ablation (QAT vs PTQ) is blocked** on `esp-ppq`, so the input mode cannot
  yet be frozen jointly with the quantization method as the plan requires.
- Thresholds here are set per run to a 5% board-level false-alarm target purely to make the
  modes comparable. The plan sets the real target from `reference`, which does not exist
  yet, and that target still needs team approval.

---

## 5. Class imbalance: no balancing needed

```bash
python eval/evaluate_run.py --run runs/control_s0 --fold val
```

Plain masked BCE does **not** collapse to predicting negatives:

| diagnostic | control, seed 0, val |
| --- | --- |
| macro AP | 0.4193 (per-class base rate ≈1.1%) |
| true positive rate | 6.48% of non-ignored entries |
| predicted positive rate at p=0.5 | 3.20% |
| recall at p=0.5 | 26.7% |
| probabilities | mean 0.061, p99 0.80, max 1.00 |

It is conservative, firing at about half the true rate, but it ranks well above chance and
makes confident positive predictions. So **class balancing does not become part of the
shared setup**, which keeps it available as a component of the contribution rather than a
baseline fix.

### Why a low absolute recall is not a failure

A board carries **25 tiles × 6 classes = 150 independent decisions**, so a 5% board-level
false-alarm budget permits only ~0.03% of them to fire. Control, seed 0, validation:

| target board FA | achieved | defect recall | precision |
| --- | --- | --- | --- |
| 1% | 0.0% | 2.11% | 80.6% |
| 5% | 5.0% | 4.80% | 71.7% |
| 10% | 10.0% | 10.63% | 71.2% |
| 20% | 20.0% | 16.16% | 66.3% |
| 50% | 50.0% | 30.79% | 57.1% |
| 90% | 90.0% | 78.02% | 24.8% |

The operating point, not the model, dominates the headline number — which is why the
false-alarm target has to be agreed before any recall figure is quoted.

---

## 6. Deployment path: ESP-DL

Verified against ESP-DL commit `5989f4a` (2026-11-02), on the PC only.

```
PyTorch -> ONNX (opset 18) -> ESP-PPQ -> .espdl -> ESP-DL -> ESP-IDF
```

| check | result |
| --- | --- |
| operators supported in int8 on ESP32-S3 | **yes** — graph is 30 nodes of `Conv`, `Relu`, `AveragePool`, `GlobalAveragePool`, `Flatten`, `Gemm` |
| `Conv` groups restriction ("1 or input_c") | satisfied — the separable blocks are fully depthwise |
| `BatchNormalization` | absent from ESP-DL's table, so it must fold; verified zero survive export |
| ONNX vs PyTorch numerics | match to 1.6e-14 (onnxruntime) |
| gate: C++ vs numpy | **bit-identical** on 200 real DeepPCB tiles, every radius |

The TFLite Micro fallback is **not needed**. ESP-DL also supplies two plan requirements as
built-ins: `export_test_values` + `dl::Model::test()` is the PC-vs-device parity check, and
`dl::Model::profile()` gives per-layer latency and a memory breakdown.

**Main remaining risk is quantization, not operator coverage.** The ESP32-S3 quantizes
**per-tensor, symmetric, with power-of-two scales** (the P4 gets per-channel Conv/Gemm; the
S3 does not), and per-tensor weight scales are the known weak point of depthwise
convolutions. One piece of luck: the pooled binary input takes values
`{0, ¼, ½, ¾, 1}`, all exactly representable at `exponent = −6`, so **input quantization is
lossless**.

Two export traps found and fixed:

* **An untrained model exports `Identity` nodes, which ESP-DL does not implement.** With
  default BatchNorm statistics every folded convolution bias is the same zero vector, ONNX
  deduplicates them, and the duplicates become aliases. A trained model exports cleanly —
  confirmed on the trained control. This matters because the plan calls for pushing an
  *untrained* student through the path first, where it would have looked like a blocker.
* **PyTorch's dynamo exporter writes weights to an `.onnx.data` sidecar**, so copying the
  `.onnx` alone to Colab would silently lose them. Both exporters now emit a single
  self-contained file.

---

## 7. Engineering measurements

| | before | after | command |
| --- | --- | --- | --- |
| data loading per tile | 16.9 ms | **0.22 ms** | `TileImageDataset` amortises image-level augmentation over 25 tiles |
| data loading per epoch | 5.6 min | **4.4 s** | |
| training, one run | — | ~7 min (22 epochs, 4 CPU cores) | `python training/train.py` |
| input ablation, 9 runs | — | 67 min | `python training/run_matrix.py --set input_ablation` |

A second data-loading bug is fixed: augmentation was driven by one shared `Generator`, so
DataLoader workers — being forked copies — all replayed the same stream, and an item's
transform depended on how many items had been drawn before it. It is now seeded from
`(seed, epoch, index)`, and a test asserts `num_workers` does not change the data.

---

## 8. Not yet measured

| | blocked on |
| --- | --- |
| PC int8 accuracy, QAT vs PTQ | `esp-ppq` (pins `python >=3.11,<3.13`) |
| on-device latency, SRAM/PSRAM, flash, energy | hardware access |
| PC-vs-device parity (`dl::Model::test()`) | hardware access |
| gate on/off latency and energy per board | hardware access |
| `reference`, `gate_only`, `reweight_only`, `hinton_kd`, `defect_kd`, `full` | the frozen configuration, pending the quantization ablation |
| teacher | trained, but not yet run |
