# PKU-Market-PCB — Step 1 inspection (Checkpoint 1)

> **Superseded in one place.** §5 recommends a global Otsu on the green
> channel. That holds for nine of the ten boards and fails outright on board
> 08, which only showed up at Step 3 when the gate was measured per board.
> The binarization that ships uses two thresholds; see
> `PKU_STEP2_REPORT.md` and `data/binarize.py`. Everything else below stands.

Branch `pku-dataset`, tag `deeppcb-v1` on `095f32b`. Nothing is committed yet.
Every number below was measured on this machine; the scripts are in `tools/`.

Dataset: [Ironbrotherstyle/PCB-DATASET](https://github.com/Ironbrotherstyle/PCB-DATASET)
(`a5769f5`), the GitHub mirror of PKU-Market-PCB. The official page gives a Baidu
Pan link only, which needs an account; the mirror has the layout the brief
describes, including `PCB_USED/`. 3.7 GB on disk, gitignored.

## Headline

The architecture carries over, but **two of the brief's assumptions are wrong**,
and one of them changes a decision in Step 2:

1. **The binarization channel must be green, not red.** Red separates pads from
   board best (Otsu bimodality 0.90 vs 0.64 for green), which is the criterion
   the brief names — but five of the six defect classes sit on traces *under the
   solder mask*, which red does not see. On red, defect visibility after erosion
   is 0–4.5% for everything except missing_hole. On green it is 87.5–100%.
2. **PKU pairs are pixel-aligned, with no shift at all.** The test images are the
   template photograph with defects composited in. So PKU closes the imaging gap
   (real color photo, real solder mask, real lighting) but adds no registration
   or lighting variation, and the defects themselves are synthetic.

Neither breaks the plan. Both change what the PKU results are evidence *for*.

## 1. What is in the dataset

| | |
| --- | --- |
| Images (non-rotated) | 693, one defect class per directory |
| Boards | 10: `01, 04, 05, 06, 07, 08, 09, 10, 11, 12` (no 02, 03) |
| Templates | 10 in `PCB_USED/`, one per board, same resolution as its test images |
| Boxes | 2,953 — mean 4.26 per image (min 1, max 6) |
| Annotation | PASCAL VOC XML; `<size>` matches the JPEG on disk in 120/120 checked |

Images per board: 01:120, 04:120, 05:60, 06:61, 07:60, 08:60, 09:60, 10:32,
11:60, 12:60. Boxes per board: 358, 358, 259, 307, 300, 298, 305, 162, 303, 303.

Image size differs **per board**, not per class: 3034×1586 (01), 3056×2464 (04),
2544×2156 (05), 2868×2316 (06), 2904×1921 (07), 2759×2154 (08), 2775×2159 (09),
2240×2016 (10), 2282×2248 (11), 2529×2530 (12).

The template name is the leading two digits of the image name
(`01_missing_hole_01.jpg` → `PCB_USED/01.JPG`), so `--split-by board` is a
one-line grouping.

## 2. Scale (px/mm)

Measured from the through-hole pad grid by autocorrelation of the pad mask,
taking the dominant spacing in the 30–200 px band as one 2.54 mm pitch. The
first attempt locked onto the ~12 px hatched-dot texture; the 30–200 px band
avoids it. Cross-checked by eye on board 01 (`eval/figures_pku/01_native_ruler.png`, grid
every 50 px): pads at ~57 px spacing, matching the computed 57 px.

| boards | pad pitch | px/mm |
| --- | --- | --- |
| 01, 04, 05, 06 | 57 px | **22.4** |
| 07, 08, 09 | 69 px | **27.2** |
| 10, 11, 12 | 103 px | **40.6** |

So PKU was shot at three different magnifications and needs a **per-board**
resample, not one global factor.

**What the code actually uses.** The pipeline runs at DeepPCB's native
resolution: `IMAGE_SIZE = 640`, `TileGrid(tile=160, stride=120)`. There is no
0.5× / 24 px/mm path anywhere in `data/`, `eval/` or `firmware/`.

**DeepPCB's own scale** is not recorded in the dataset, so I measured it off
pad geometry: an isolated through-hole pad is ~105 px across with a ~35 px
hole, i.e. ~2.0 mm and ~0.8 mm, giving **45–52 px/mm**. The brief's ~48 px/mm
is right. (Note: DeepPCB polarity is copper = *black*; `data/deeppcb.py`'s
docstring says copper is white. Harmless for XOR, but the docstring is wrong.)

**Recommendation: do not resample PKU to 48 px/mm.** That means upsampling by
1.2–2.1×, which invents no detail, and it breaks the tile geometry — see §3.
Resample instead to one common scale across boards. The right target is the
camera's px/mm at the 82 mm stand height, which is what the demo has to match;
22.4 px/mm (the scale three boards already have) is the natural choice, needing
0.82× for boards 07–09 and 0.55× for 10–12.

## 3. Defect size, and why the scale choice matters

Box short side, converted to mm with each board's own px/mm:

| class | n | short (mm) | long (mm) | short p90 (mm) | short (native px) |
| --- | --- | --- | --- | --- | --- |
| missing_hole | 497 | 2.36 | 2.58 | 2.94 | 61 |
| mouse_bite | 492 | 1.82 | 2.58 | 2.54 | 51 |
| open_circuit | 482 | 1.55 | 1.88 | 2.41 | 45 |
| short | 491 | 2.58 | 3.12 | 3.97 | 69 |
| spur | 488 | 1.96 | 3.10 | 2.67 | 54 |
| spurious_copper | 503 | 1.80 | 3.79 | 2.54 | 50 |
| **all** | 2,953 | **1.96** | | | |

DeepPCB, for comparison: 31 px short side at ~48 px/mm = **0.65 mm**. PKU
defects are about **3× larger physically** — they are drawn at component scale,
not trace scale.

This is the quantitative reason not to match DeepPCB's px/mm. A 160 px tile:

| scale | tile covers | median defect | p90 `short` defect |
| --- | --- | --- | --- |
| DeepPCB, 48 px/mm | 3.33 mm | 94 px (59% of tile) | 190 px — **larger than the tile** |
| PKU native, 22.4 px/mm | 7.14 mm | 44 px (27% of tile) | 89 px (56%) |

At 48 px/mm the biggest shorts cannot fit in any tile, so the 70% rule could
never label them and defect-level recall would be capped by geometry — the same
failure Finding 1 in `Findings.md` warns about. At ~22 px/mm the ratio of defect
to tile is close to DeepPCB's own. Keep 160/120; change the resample target.

## 4. Alignment

20 pairs sampled across all boards and classes, ±8 px integer shift search on a
600 px central crop, minimizing XOR — the firmware's method.

- Best shift: **dx = dy = 0 on all 20 pairs**. Median |shift| 0 px, max 0 px.
- XOR density at zero shift: median 0.0041 (0.41% of pixels), unchanged by the
  search.

The test images are the template photo with defects composited in, so the pair
is registered exactly and the only differences are the defects plus JPEG noise.

Consequences, both worth stating in the paper:

- The firmware's shift search cannot be *evaluated* on PKU. Misalignment
  robustness still has to come from the augmentation (`±few px`) and from the
  printed-board capture in Step 5.
- The gate's job is much easier here than on DeepPCB, where a 1-px JPEG halo on
  every trace edge put ~165 raw XOR pixels in a *clean* tile. PKU's clean
  background after a 3×3 erosion is ~35 px per 160×160 tile. A PKU gate number
  is therefore **not** comparable to the DeepPCB gate number (88% skip at 98%
  recall); only within-dataset comparisons are fair.

## 5. Binarization: channel choice, measured on defect visibility

Otsu bimodality (between-class / total variance) on the 10 templates — the
brief's criterion, "copper/laminate contrast":

| channel | gray | R | G | B |
| --- | --- | --- | --- | --- |
| mean | 0.728 | **0.899** | 0.636 | 0.830 |

By that criterion red wins. But what a global Otsu finds on these boards is the
**pads**, not the copper: the bright class is 9–22% of the image, against 60%
copper in DeepPCB. Traces are under green mask and do not survive thresholding —
visible in `eval/figures_pku/01_photo_gray_r.png` (photo | gray-Otsu | R-Otsu:
both binarizations are pad maps).

Since five of six classes are trace defects, the criterion that matters is
defect contrast, not board bimodality. Share of annotated defects with ≥1 pixel
surviving a 3×3 erosion, 20 images per class (~88–91 boxes per class):

**Red channel** — threshold from the template, XOR, erode:

| | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper |
| --- | --- | --- | --- | --- | --- | --- |
| binarize+XOR | 100% | 2.2% | 4.5% | 1.1% | 1.1% | 0.0% |
| \|diff\| > 30 | 100% | 4.4% | 8.0% | 6.9% | 3.3% | 0.0% |

**Green channel:**

| signal | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper | noise px / 160² tile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binarize+XOR | 100% | 94.4% | 87.5% | 89.7% | 94.5% | 94.2% | 34.5 |
| \|diff\| > 8 | 100% | 100% | 100% | 100% | 100% | 100% | 58.0 |
| \|diff\| > 16 | 100% | 100% | 100% | 100% | 100% | 100% | 47.1 |
| \|diff\| > 25 | 100% | 100% | 100% | 96.7% | 100% | 100% | 38.3 |
| \|diff\| > 30 | 100% | 98.9% | 98.9% | 92.3% | 97.7% | 100% | 34.4 |

Raw contrast inside the boxes confirms where the signal is — median over boxes
of max |test − template| per channel: green 56–62 for every class, red only
16–32 except missing_hole (172).

**Recommendation for Step 2:** green channel, and keep binarize-then-XOR as the
primary path (87.5–100%, same structure as the firmware, no gate rewrite). Carry
`|diff| > 16` as the measured fallback — it reaches 100% on all six classes at a
~1.4× higher noise floor. The decision between them belongs at Checkpoint 2,
where the gate's skip-rate-vs-recall curve exists.

## 6. Class mapping

PKU `missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper`;
DeepPCB `open(1), short(2), mousebite(3), spur(4), spurious copper(5),
pin-hole(6)`. Five match by name. Agreed with the brief: keep the 6-output head
and let **slot 6 = missing_hole** in PKU mode, with the class list taken from
the dataset mode rather than hardcoded. Do not merge missing_hole with pin-hole:
a pin-hole is a void *in* copper, a missing hole is an undrilled pad, and in
these photos they look nothing alike.

## 7. State of the repo

- `deeppcb-v1` tags `095f32b`, the newest commit holding DeepPCB results — the
  tip of the unmerged `claude/lucid-cannon-eyj5yg`, not `main`. `main` has only
  the Nov-3 merge plus your brief upload.
- `pku-dataset` branches from `095f32b` with `main` merged in, so it carries the
  Nov-4 teacher/ablation work *and* the brief.
- Existing tests on this branch: **197 passed, 8 skipped**, no failures. The
  remaining skips are environmental: no host C++ compiler (3 gate-parity tests,
  so firmware↔Python gate parity is unverified on this machine) and no
  `onnxscript` (2 dynamo-exporter tests); the rest needed the tile dataset.
- The DeepPCB tile dataset rebuilt cleanly from the pinned clone
  (`08e98c4`): 1,500 pairs → `data/tiles/deeppcb_tiles_t160_s120.npz`, 53–55%
  clean tiles per fold, matching the committed numbers.
- Uncommitted: `.gitignore` (adds `data/PKU-DATASET/`, `data/tiles_pku/`),
  `tools/*.py` (the six inspection scripts), `eval/figures_pku/*.png`, and this
  report. No pipeline file has been touched.

## Open question for Checkpoint 1

The scale target. §2 recommends 22.4 px/mm (and *not* DeepPCB's 48), but the
number that actually matters is what the ESP32 camera sees at the 82 mm stand
height. If you can capture one frame of a ruler or a 2.54 mm header at that
height, I will set the resample factor from it instead of from the dataset.
