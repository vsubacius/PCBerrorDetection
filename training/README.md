# training/

Model definitions, losses and train loops.

| File | State |
| --- | --- |
| `models.py` | `TileStudent` — **done**, frozen and tested |
| `losses.py` | masked BCE / reweighted BCE / defect-aware KD — not started |
| `train.py` | teacher, control, KD and QAT loops — not started |

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

## Planned

* `losses.py` — one masked loss module covering plain BCE, reweighted BCE and the
  **defect-aware distillation loss**, the project's novel contribution:

  ```
  L = mean_tiles  w(tile) * [ BCE(sigmoid(z_s), y)
                              + lambda * T^2 * BCE(sigmoid(z_s/T), sigmoid(z_t/T)) ]
  ```

  with `w = 1` for clean tiles and `alpha > 1` for defect tiles, times the inverse
  class-frequency weight from `TileData.class_pos_weights()`, normalised to mean 1 per batch.
  Setting `w = 1` everywhere reduces it to the sigmoid-adapted Hinton KD used as the
  prior-method comparison, so both live in one implementation. The `T^2` factor is included
  (the repo formula omitted it) so that `lambda` does not have to absorb the temperature;
  the same form is used in both KD runs.

  The loss must honour the per-tile **ignore mask** — a channel marked ignored contributes
  nothing to either term. `TileData` yields `(x, y, ignore)` for exactly this reason.

* `train.py` — loops for the ResNet-18 teacher, the control student, the KD students, and QAT.

### Known issues to fix before the first real training run

* `TorchTileDataset.__getitem__` augments the whole 640×640 pair and cuts all 25 tiles just
  to return one, so it does ~25× redundant work per sample. A full-board morphology op per
  *tile* access will dominate epoch time.
* Its `self.rng` is shared mutable state, so DataLoader workers fork identical generator
  states and emit identical augmentation streams. Needs per-worker re-seeding.
