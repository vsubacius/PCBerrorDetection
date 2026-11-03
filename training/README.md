# training/

Model definitions, losses and train loops.

| File | State |
| --- | --- |
| `models.py` | `TileStudent` — **done**, frozen and tested |
| `losses.py` | masked BCE / reweighted BCE / Hinton KD / defect-aware KD — **done** |
| `teachers.py` | ResNet-18 distillation teacher — **done** |
| `train.py` | one train loop for every run — **done** |
| `run_matrix.py` | resumable sweep driver — **done** |
| QAT | blocked on esp-ppq (`python >=3.11,<3.13`) |

## `TileStudent` (frozen at checkpoint 1, 2026-11-03)

The Stage-2 classifier every run in the experiment matrix shares. Only `in_channels`
changes across the input-mode ablation (A=1, B=2, C=2).

```
AvgPool2d(2)                 160x160xC -> 80x80xC
Conv3x3 s2 -> 16, BN, ReLU      80x80  -> 40x40x16
DSBlock  16 -> 32   s2          40x40  -> 20x20
DSBlock  32 -> 64   s2          20x20  -> 10x10
DSBlock  64 -> 96   s2          10x10  ->   5x5
DSBlock  96 -> 128  s2            5x5  ->   3x3
DSBlock 128 -> 128  s1            3x3  ->   3x3
DSBlock 128 -> 128  s1            3x3  ->   3x3
GlobalAvgPool + Linear(128, 6)        -> 6 logits
```

Measured with `python training/models.py --layers` (MACs counted from observed tensor
shapes via forward hooks, so the number cannot drift from the model):

| | mode A (1 ch) | modes B/C (2 ch) | team target |
| --- | --- | --- | --- |
| parameters | 60,966 | 61,110 | ~63K |
| int8 weights | 57.5 KB | 57.6 KB | ~63 KB |
| MACs / tile | 1.33 M | 1.56 M | ~1.5 M |
| MACs / board | 33.2 M | 39.0 M | — |
| peak activation | 31.2 KB | 37.5 KB | 16 KB |
| + input tile | 31.2 KB | 62.5 KB | — |

**The 16 KB activation target is not met and was not reachable.** A 160×160 int8 tile is
25 KB of input buffer on its own, so the only way to hit 16 KB is to halve the tile
resolution. Against 512 KB of SRAM and 8 MB of PSRAM that target is not binding, so the
design spends activation memory and keeps resolution. Params and MACs are met.

### Design notes

* **The 4× reduction happens before the stem convolution.** A stride-2 stem on the full tile
  would compute at 80×80 and cost 0.92 M MACs by itself; pooling first costs 0.23 M for the
  same output resolution.
* **Average pooling, not max.** Copper covers ~81% of a DeepPCB board, so a 2×2 OR saturates
  to all-ones and destroys the signal. A 2×2 mean gives 5 levels and is still multiply-free
  on the MCU (3 adds and a shift).
* **The reduction is safe for this dataset.** No annotated defect box is thinner than 23 px
  on either axis (p1 of the min-dimension over all 10,013 boxes), so every box still spans
  ≥ 5 px at 40×40.
* **The head returns logits, never probabilities.** The losses use the `*_with_logits`
  forms and the firmware compares against `log(p/(1-p))`, which keeps a quantized sigmoid
  out of the int8 graph.
* **Convolutions carry no bias** so that BatchNorm folds cleanly at export — ESP-DL has no
  `BatchNormalization` kernel.

[`tests/test_models.py`](../tests/test_models.py) pins the parameter count, the MAC count and
the spatial ladder. The matrix is only interpretable if every run shares one student, so a
change there should fail loudly rather than silently invalidate earlier results.

## Running a job

Every run is fully described by a `TrainConfig`, written to `runs/<id>/config.json`.
Reproducing a run from its config gives **bit-identical** numbers.

```bash
python training/train.py --run-id control_s0 --loss plain --representation A --seed 0
```

```bash
python training/train.py --config runs/control_s0/config.json
```

```bash
python training/run_matrix.py --set input_ablation --dry-run
```

`run_matrix.py` treats a run with a `metrics.json` as finished, so a sweep resumes
instead of restarting — a restarted worker had already killed two trainings mid-sweep.
Runs go one at a time on purpose: concurrent jobs on four cores thrash, and they make
wall-clock timings incomparable between runs.

Checkpoints are selected on **validation macro average precision**, which is
threshold-free. Selecting on recall at a fixed threshold would entangle model selection
with the threshold choice, and thresholds are only set afterwards, on validation int8
outputs, to hit an agreed board-level false-alarm rate.

## `losses.py`

One masked loss covers all four training configurations, so the runs differ in
configuration and never in code:

| kind | `alpha` | teacher | role |
| --- | --- | --- | --- |
| `plain` | 1 | no | the control and `reference` |
| `reweighted` | >1 | no | tests whether weighting alone explains a gain |
| `hinton_kd` | 1 | yes | the prior-method comparison |
| `defect_kd` | >1 | yes | **the novel contribution** |

```
L = mean_(tile, class)  w(tile, class) * [ BCE(sigmoid(z_s), y)
      + lambda * T^2 * BCE(sigmoid(z_s / T), sigmoid(z_t / T)) ]
```

Three decisions the repo had left open, all now settled and tested:

* **`T^2` is included**, identically in both KD runs. For this sigmoid form the
  distillation gradient scales as `1/T^2` — the same asymptotics as Hinton's softmax
  argument — so without the factor `lambda` would have to absorb the temperature and the
  `lambda`/`T` sweep would be unreadable. A test shows the gradient scale is flat in `T`
  with the factor and collapses by >20× without it.
* **Weights are normalised to mean 1 per batch.** Otherwise `alpha` would change the
  effective learning rate, and `reweight_only` would differ from `reference` in step size
  as well as in weighting — confounding the very thing it is meant to isolate.
* **`plain` and `hinton_kd` refuse `alpha != 1`.** Those kinds are *defined* by `w = 1`,
  so silently accepting a weight would make two rows of the run matrix identical.

The ignore mask is honoured in **both** terms: an ignored `(tile, class)` entry
contributes nothing, and the mean is over surviving entries so the loss scale does not
depend on how much was ignored.

Hinton KD was defined over softmax outputs. This sigmoid multi-label form is an
**adaptation** and the paper must describe it that way.

## `teachers.py`

ResNet-18, 11.17M parameters and 885M MACs against the student's 61K and 1.33M. Adapted:
input channels follow the frozen input mode (the plan had fixed it at 2 channels, which
would have let a KD run distil from a teacher seeing inputs the student never gets); six
sigmoid logits; no pretrained weights, since ImageNet initialisation is of little use on a
binarized PCB tile and `download.pytorch.org` is unreachable here anyway.

Unlike the student, the teacher sees the tile at **full 160×160 resolution** — it reaches
the same 40×40 feature map but computes it from every pixel rather than a 4×-pooled copy.
That is much of why it should be worth distilling from.

Caveat: 11M parameters on 20,000 tiles drawn from 800 images — themselves only 7 physical
boards — will overfit hard. **Check the teacher actually beats the student on validation
before reading anything into a KD result.**

## Still planned

* QAT, once esp-ppq is available on a `<3.13` interpreter.

  ```
  L = mean_tiles  w(tile) * [ BCE(sigmoid(z_s), y)
                              + lambda * T^2 * BCE(sigmoid(z_s/T), sigmoid(z_t/T)) ]
  ```

### Data loading, fixed

Two problems in `TorchTileDataset` are resolved, and both mattered:

* It augmented the whole 640×640 pair and cut all 25 tiles just to return one — **16.9 ms
  per tile, or 5.6 minutes of data loading per epoch.** `TileImageDataset` serves one
  image's tiles per item, amortising that work: **0.22 ms per tile, 4.4 s per epoch.**
* Its shared `Generator` was wrong twice over: DataLoader workers are forked copies, so
  every worker replayed the same augmentation stream, and an item's transform depended on
  how many items had been drawn before it. Augmentation is now seeded from
  `(seed, epoch, index)`, and a test asserts that `num_workers` does not change the data.
