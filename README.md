# PCB Defect Detection on an ESP32-S3

A **reference-guided, gated tile cascade with defect-aware distillation** for bare-PCB
inspection on a microcontroller. Course project for ECE 690OD (On-Device ML, UMass Amherst,
Fall 2026). Target hardware: Seeed XIAO ESP32-S3 Sense.

The idea: cut an aligned (template, test) PCB image pair into overlapping tiles, use a very
cheap Stage-1 *gate* to throw away tiles that match their template, and run an int8 multi-label
CNN only on the tiles that survive. The student CNN is trained with a **defect-aware
distillation loss** that weights defect tiles above clean ones — that loss is the project's
novel contribution.

See [`CLAUDE.md`](CLAUDE.md) for the full project plan, deliverables and timeline.

## Status

| Milestone | State |
| --- | --- |
| Tiling + label pipeline | **done** — `data/`, [`notebooks/01_dataset_and_tiles.ipynb`](notebooks/01_dataset_and_tiles.ipynb) |
| Tile-grid choice validated against defect sizes | **done** — 160/120 confirmed, see below |
| Stage-1 classical gate characterised | **done** — `eval/gate_analysis.py`, see below |
| Dummy model on the ESP32-S3 | not started |
| Teacher + control student, QAT | not started |
| Defect-aware KD, learned gate, FOMO/Hinton comparisons | not started |

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
python data/build_tiles.py --download          # clone DeepPCB (~221 MB) and build the tiles
```

```bash
python data/grid_sweep.py                      # defect sizes and tile-grid trade-offs
```

```bash
python eval/gate_analysis.py                   # how separable clean vs defective tiles are
```

```bash
python -m pytest tests -q                      # unit tests for the labelling rule
```

Or open [`notebooks/01_dataset_and_tiles.ipynb`](notebooks/01_dataset_and_tiles.ipynb) in Colab
straight from GitHub — it does the download, the plots and the build, and needs no GPU.

## Layout

```
data/      download, tiling, labelling, dataset loader   (no raw data committed)
notebooks/ Colab notebooks
training/  model definitions, losses, train loops         (empty)
export/    quantization + ESP-DL / TFLM conversion        (empty)
firmware/  ESP-IDF project for the XIAO ESP32-S3          (empty)
eval/      metrics, plots, on-device log parsing
paper/     4–6 page workshop-style write-up               (empty)
```

Core modules, all importable from `data/`:

| Module | Purpose |
| --- | --- |
| `deeppcb.py` | download, index and load DeepPCB pairs; class ids and channel order |
| `tiling.py` | `TileGrid` geometry, the tile-labelling rule, gate labels |
| `build_tiles.py` | builds the `.npz` tile dataset with an image-level train/val split |
| `tile_dataset.py` | loads a fold, cuts tiles on the fly, input reps A/B/C, augmentation |
| `grid_sweep.py` | defect-size stats and the tile/stride trade-off table |
| `eval/gate_analysis.py` | Stage-1 classical gate separability vs misalignment |

## Dataset

[DeepPCB](https://github.com/tangsanli5201/DeepPCB): 1,500 aligned (defect-free template,
defective test) pairs, 640×640, binarized, with box annotations for 6 defect types. **Research
use only.** Never committed here — `data/build_tiles.py --download` fetches it.

Verified against the real clone (1,500 pairs, 10,013 boxes, 11 groups):

* The official split is exactly **1,000 trainval / 500 test**, and every file the split lists
  reference exists.
* The split lists name `<group>/<id>/<name>.jpg`, which is *not* a real file — the images on
  disk have `_test` / `_temp` appended to the stem.
* One template, `group90100/90100/90100034_temp.jpg`, has no test-image partner and is in
  neither split list. It is skipped.
* At least one JPEG is stored as RGB, not grayscale, so everything is converted to mode `L`.
* The images were binarized *before* JPEG compression, so about 1.2% of pixels come back
  mid-gray. Everything here re-thresholds at 127 to a true `{0,1}` image. This matters: see
  the gate result below.
* Defect density is ~6.7 per image (1–15), far higher than a real board. Gate savings should
  therefore also be reported on the 500 test templates run as defect-free boards.

## Tile grid: 160 px tiles, stride 120 (40 px overlap)

A defect is only findable if some tile holds at least `pos_thresh` (0.70) of its area —
otherwise no tile carries its label and the cascade cannot see it, whatever Stage 2 does. That
makes **reachability a hard ceiling on defect-level recall**, set purely by the geometry.

Measured over all 10,013 boxes (`python data/grid_sweep.py --split all`, abridged):

```
 tile/stride    grid  ovl  tiles  px cost  reach@0.70    full   clean    pos    ign
-----------------------------------------------------------------------------------
     128/128     5x5    0     25    1.00x     78.76%   53.9%   66.6%  20.6%  17.4%
     160/160     4x4    0     16    1.00x     82.37%   62.6%   56.8%  30.8%  19.6%
     160/120     5x5   40     25    1.56x     99.95%   95.4%   53.8%  32.0%  22.3%
      160/96     6x6   64     36    2.25x     99.99%   98.7%   52.2%  33.5%  22.6%
     192/112     5x5   80     25    2.25x    100.00%   99.6%   42.1%  44.9%  26.7%
     224/104     5x5  120     25    3.06x    100.00%  100.0%   32.6%  57.2%  28.3%
```

**The proposed 160/120 grid holds up.** Only 5 of 10,013 defects (0.05%) are unreachable, so
the recall ceiling is 99.95%; the five are the largest `open` and `short` defects (121–173 px).
Defects have a median max-dimension of 38 px and a p99 of 101 px, so the 40 px overlap is *not*
≥ nearly all defect sizes as the plan assumed — but the 70%-area rule rescues it, because a
defect only needs most of itself inside one tile, not all of it. Going wider buys ~0.05% recall
for 1.4× the Stage-2 pixels, which is not worth it.

Two decisions this raises, both worth settling before the mid-project presentation:

1. **The evaluation metric in the plan is stricter than the training label.** It counts a defect
   as found only if *a tile fully containing it* predicts its class, and only 95.4% of defects
   are fully inside any tile at 160/120 — so that metric caps recall at 95.4% for a geometric
   reason unrelated to the model. Scoring against the best-containment tile (the same ≥70% rule
   used for labels) raises the ceiling to 99.95% and measures the model instead of the grid.
2. **`split_by`.** The plan splits train/val by image, which `build_tiles.py` does by default.
   DeepPCB's 1,500 images come from only 11 physical boards, so an image-level split still lets
   train and val share a board. `--split-by group` is available as a stricter check; expect the
   val score to drop, and report both if they differ.

Label population at 160/120 (37,500 tiles): 53% clean, 30% carry a positive class, 24% hold
only a fragment and are ignored. Inverse-frequency positive weights land between 11.6 and 17.0,
so the classes are reasonably balanced — the real imbalance is positive-vs-negative, not
class-vs-class.

## Stage-1 gate: count the XOR, but denoise it first

The obvious classical gate is `count(XOR(test, template)) > threshold`. Measured on the val
fold, clean vs defective tiles (`python eval/gate_analysis.py`):

```
 feature  shift  med clean  med defect     AUC  thr@98  skip@98  recall@skip50
     raw    0px        165         483  0.7760     129    43.6%          94.5%
     raw    3px        457        1045  0.7148     166    31.1%          84.8%
  erode1    0px          0          83  0.9530       5    88.1%          99.9%
  erode1    3px          9         183  0.7773      11    51.0%          98.4%
  erode2    0px          0          21  0.9243       0    98.4%          86.9%
  erode2    3px          0          27  0.8886       0    90.3%          88.2%
```

`skip@98` is the share of clean tiles the gate skips while keeping 98% of defect tiles — i.e.
the Stage-2 compute actually saved. `shift` jitters the template by ±N px to stand in for
residual registration error.

**Raw XOR counting is a weak gate (AUC 0.78) and it is not the misalignment that breaks it.**
The median *clean* tile already carries 165 XOR pixels, because re-binarizing JPEG-compressed
images leaves a 1-px fringe of disagreement along every copper edge, and that fringe swamps
small defects. Eroding the XOR mask before counting removes the fringe: a 3×3 erosion takes AUC
to 0.95 and `skip@98` from 44% to 88%, and a 5×5 erosion is the most misalignment-tolerant
(AUC 0.89 and 90% of clean tiles skipped even at ±3 px). A *k*×*k* binary erosion is a sliding
AND, separable into two 1-D passes with no multiplies, so it is cheap on the ESP32-S3.

Consequences for the project:

* The classical gate in the plan should be `count(erode(XOR)) > threshold`, not
  `count(XOR) > threshold`. Keeping both as a gate ablation is worth a row in the paper, since
  the difference is large and the cost is tiny.
* The learned micro-CNN gate now has to beat a *strong* baseline (~90–98% of clean tiles skipped
  at 98% tile recall), not a weak one. That is a better result either way, but it changes what
  the learned gate has to prove.
* These numbers are tile-level recall on pre-registered pairs, not end-to-end defect recall.
  They set up the gate-threshold sweep (energy per board vs. defect recall) but do not replace
  it.

## Conventions

* Class order is fixed everywhere: `open, short, mousebite, spur, copper, pin-hole`
  (dataset ids 1–6 → channels 0–5). Use `deeppcb.CLASS_IDS` / `CLASS_ID_TO_CHANNEL`.
* Boxes and tiles are **half-open** `[x1, x2) × [y1, y2)`, so a tile at `(ox, oy)` is exactly
  `img[oy:oy+tile, ox:ox+tile]`.
* Images are `{0, 1} uint8` with copper = 1 (white), covering ~82% of a board.
* The built `.npz` bit-packs the images along x, so the whole dataset is ~10 MB and caches
  easily on Drive. Tiles are cut on the fly, which also means the tile-size ablation needs no
  rebuild — `TileData.load(path, fold, grid=TileGrid(192, 112))` relabels from the stored boxes.
* `ignore` masks are never set on a channel that is also positive, so a masked loss can use `y`
  directly.

## Reproducing the numbers in this README

```bash
python data/build_tiles.py --download && python data/grid_sweep.py --split all && python eval/gate_analysis.py
```

Figures land in `eval/figures/` (gitignored); the notebook writes
`eval/figures/defect_size_hist.png`.
