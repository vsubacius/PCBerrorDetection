# PKU-Market-PCB — Steps 2 and 3 (Checkpoint 2)

Branch `pku-dataset`. DeepPCB results are tagged `deeppcb-v1` (`095f32b`) and
nothing on that path changed.

## What was built

| | |
| --- | --- |
| `data/binarize.py` + `firmware/pcb_bench/main/pcb_binarize.{hpp,cpp}` | one binarization, written twice, integer-only |
| `tests/test_binarize.py`, `tests/test_binarize_parity.py` + harness | 47 behaviour tests; the parity test compiles the C++ and compares |
| `data/pku.py` | PKU photographs → DeepPCB-format pairs |
| `data/build_tiles.py --dataset pku`, `--split-by board` | tiles from either dataset |
| `data/grid_sweep.py --dataset pku` | the geometry check on either dataset |
| `eval/pku_sanity_sheet.py` | the visual check |

`eval/gate.py`, `tiling.py`, `tile_dataset.py`, `training/` and the firmware
gate are untouched: the converter writes DeepPCB's own on-disk shape, and
`pku.py` exposes `deeppcb.py`'s interface, so the switch is one module.

## The result that changed the design

A single global Otsu on green — what Checkpoint 1 recommended — **fails
completely on board 08**, and only on board 08. Measured per board on the
converted tiles, with the classical gate (3×3 erosion, radius 1):

| board | fold | defect tiles | scoring zero | median score | clean tiles at zero |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | train | 635 | 0.6% | 120 | 95.0% |
| 04 | train | 427 | 0.0% | 447 | 99.1% |
| 05 | train | 348 | 0.0% | 385 | 99.5% |
| 07 | train | 380 | 0.0% | 280 | 99.7% |
| 09 | train | 479 | 0.0% | 231 | 99.4% |
| 10 | train | 256 | 0.0% | 185 | 99.3% |
| 12 | train | 918 | 0.0% | 86 | 99.7% |
| **08** | **val** | **521** | **83.1%** | **0** | 99.9% |
| 06 | test | 407 | 0.0% | 327 | 99.6% |
| 11 | test | 408 | 1.2% | 145 | 99.3% |

A defect tile scoring exactly zero cannot be recovered by any threshold, so
board 08 capped the gate's ceiling at 16.5% — and board 08 was the whole
validation fold, which is how a dataset-wide conclusion nearly came out of one
board.

The cause is not exposure (board 08's test images and its template have
identical mean green, 89.2) and not the resample (it fails at native scale
too). Its green histogram puts Otsu at 106, *inside* the copper distribution
instead of below it, so traces are classified as laminate and a defect on a
trace produces no XOR at all.

Three candidates, measured on 99–101 boxes per board, as the share of defects
with at least one pixel surviving the gate's erosion:

| binarization | board 08 | board 01 | board 11 | noise, eroded px per 160² tile |
| --- | ---: | ---: | ---: | --- |
| global Otsu on green | **4%** | 100% | 100% | 0.0 / 2.1 / 11.0 |
| flat-field, then green | 100% | **72%** | 100% | 5.6 / 1.8 / 8.8 |
| **two thresholds (shipped)** | **100%** | **100%** | **100%** | 6.0 / 2.0 / 10.1 |

The shipped rule is

```
copper = (red > t_red) | ((green > t_green) & (red <= t_red))
```

Red finds exposed metal — tinned pads read ~126 in red against ~5 for green
solder mask. Green, thresholded over the masked pixels only, then finds copper
*under* the mask. It costs one extra histogram pass and one compare per pixel
on the chip. Flat-fielding stays in the code for camera frames, where the
illumination gradient is real, but it is off for dataset photographs — it
rescues board 08 and then breaks board 01.

## Conversion

693 photographs → **5,267 crops**, 640×640, binary, in DeepPCB's on-disk shape.

- **Scale**: per board to 22.4 px/mm (57 px per 2.54 mm); boards 07–09 by
  0.826, boards 10–12 by 0.553, boards 01/04/05/06 unchanged.
- **Boxes**: 2,953 of 2,953 kept. A defect that no grid crop holds to 70% gets
  a crop centred on it, so the grid drops nothing.
- **Folds by board**: train 01/04/05/07/09/10/12, val 08, test 06/11. No board
  appears in two folds — 60-odd photographs of one board share a layout, so
  any other split leaks it.
- **Alignment**: the shift search runs and finds (0, 0) almost everywhere.

## Checks, after the fix

### Defect visibility, per class

Share of defects with at least one pixel surviving the gate's 3×3 erosion, over
every val and test crop (`data/tiles_pku`, 1,067 boxes):

| class | boxes | survives | median px |
| --- | ---: | ---: | ---: |
| copper | 179 | 100.0% | 328 |
| missing_hole | 179 | 100.0% | 195 |
| mousebite | 179 | 100.0% | 213 |
| open | 173 | 100.0% | 172 |
| short | 181 | 100.0% | 143 |
| spur | 176 | 99.4% | 214 |
| **all** | **1,067** | **99.9%** | |

DeepPCB on the same measurement: 99.7% overall, 98.9–100% per class, median
25–102 px. No class is lost, and none is close to being lost.

### Per board, after the fix

Defect tiles scoring exactly zero through the classical gate — the number that
caps recall — before and after the binarization change:

| board | fold | defect tiles | zero (global green) | zero (shipped) | median score |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | train | 635 | 0.6% | 0.3% | 120 |
| 04 | train | 427 | 0.0% | 0.0% | 447 |
| 05 | train | 348 | 0.0% | 0.0% | 380 |
| 07 | train | 380 | 0.0% | 0.0% | 272 |
| 09 | train | 479 | 0.0% | 0.0% | 222 |
| 10 | train | 256 | 0.0% | 0.0% | 180 |
| 12 | train | 918 | 0.0% | 0.0% | 86 |
| **08** | val | 521 | **83.1%** | **0.2%** | 184 |
| 06 | test | 407 | 0.0% | 0.0% | 325 |
| 11 | test | 408 | 1.2% | 0.0% | 153 |

### Tile reachability (`data/grid_sweep.py --dataset pku`)

At the current 160/120 grid, unchanged:

| | PKU | DeepPCB |
| --- | ---: | ---: |
| reach @ 0.70 (the recall ceiling) | **99.31%** | 99.95% |
| fully inside some tile | 78.3% | 95.4% |
| clean tiles | 93.6% | 53.8% |
| per-class reach | 98.1–100% | — |

Wider grids do better (160/96 → 99.67%, 192/112 → 99.97%) at 1.4–1.4× the
Stage-2 pixel cost. **Not recommended**: 99.31% is already above the point
where geometry is the limiting factor, and changing the grid would make the
two datasets incomparable, which constraint 2 forbids.

The gap between 99.31% reach and 78.3% full containment is larger here than on
DeepPCB because PKU defects are physically ~3× bigger. That makes Finding 1's
recommendation — score against the same ≥70% tile used for training, not full
containment — more urgent, not less: on PKU a full-containment metric would
cap a perfect model at 78.3%.

### Stage-1 gate (`eval/gate_analysis.py`, radius 1)

Threshold chosen on validation at a 98% defect-tile recall target, then applied
unchanged to the test fold:

| | threshold | val recall | val skip | **test recall** | **test skip** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **PKU** | 38 | 98.1% | **99.4%** | **100.0%** | **98.5%** |
| DeepPCB | 4 | 98.6% | 87.2% | 98.8% | 83.3% |

AUC on PKU val is 0.9974 with the erosion, against 0.9526 without it — the
erosion matters here too, though less dramatically than on DeepPCB (0.95 vs
0.78), because PKU has no JPEG fringe along every trace edge.

**This number is not comparable to DeepPCB's, and should not be reported as an
improvement.** PKU's test images are the template photograph with defects
composited in, so the pair is registered exactly and a clean tile is genuinely
empty. The gate looks near-perfect because the dataset is easy, not because
anything got better. Under simulated misalignment the advantage disappears:

| jitter | 0 px | 1 px | 2 px | 3 px |
| --- | ---: | ---: | ---: | ---: |
| recall @ threshold | 98.1% | 98.1% | 98.3% | 98.3% |
| clean skip | 99.4% | 99.4% | 93.3% | 56.5% |

Radius 2 caps at a 97.5% ceiling and cannot reach the 98% target at any
threshold — the same conclusion the DeepPCB analysis reached, independently
confirmed. Radius 1 stays.

### Visual check

`eval/figures_pku/pku_sanity_sheet.png` — 10 defect crops and 2 clean ones, as
template | test with boxes and the tile grid | eroded XOR. The XOR is empty
everywhere except inside the annotated boxes, and the clean crops are entirely
empty (0 surviving pixels), which is the single best evidence that the scale
factor, the board-to-template pairing and the binarization are all right.

## What I did not do

- **Parity is unverified on this machine.** `tests/test_binarize_parity.py` and
  the three existing gate-parity tests skip here: there is no host C++
  compiler. The firmware binarization is therefore *written* to match and
  *untested* against the Python. Installing MinGW or LLVM would run them.
- No training. Step 4 is next and needs your go-ahead.
- `eval/gate.py`, `tiling.py`, `tile_dataset.py` and `training/` are unchanged;
  `build_tiles.py` and `grid_sweep.py` gained a `--dataset` switch only.
- Tests: **258 passed, 17 skipped**, no failures. The DeepPCB tile dataset and
  its gate numbers reproduce exactly.

## Recommendation

Proceed to Step 4 on this data, with two things written into the plan now:

1. Report PKU and DeepPCB gate numbers side by side but never subtract them.
   PKU's 98.5% skip measures a dataset with no registration error; DeepPCB's
   83.3% measures one with a JPEG fringe on every edge. The honest headline is
   the misalignment sweep above, which is the only PKU number that speaks to
   the live demo.
2. Keep 160/120 and the ≥70% metric on both datasets.

