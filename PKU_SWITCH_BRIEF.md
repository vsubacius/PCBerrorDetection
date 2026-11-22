# Brief for Claude Code: add PKU-Market-PCB next to DeepPCB, keep the architecture

Repo: `github.com/vsubacius/PCBerrorDetection` (read `CLAUDE.md`, `Findings.md`, `data/`, `eval/gate.py` first).

## Goal

The live demo must work on a **printed color photo of a real PCB** or a **real PCB**, not only on a printed
black-and-white DeepPCB image. To close that domain gap, add the PKU-Market-PCB dataset (real photos of bare
PCBs, 10 defect-free templates, six defect types).

**Hard constraints**

1. Do not change the project idea. The pipeline stays: template/test pair → 640×640 binary image →
   5×5 grid of 160 px tiles, stride 120 → Stage 1 XOR + 3×3 erosion gate → Stage 2 6-output student →
   defect-aware KD.
2. The DeepPCB path must keep working exactly as it does now. All existing tests must pass with no change.
   The team must be able to show the old method side by side with the new one.
3. Do not edit or retrain anything until I approve each checkpoint below.

## Core idea: make PKU look like DeepPCB, then reuse everything

Write one converter, `data/pku.py`, that turns PKU photos into **DeepPCB-format pairs**:
640×640 binary, aligned `(template, test)` crops plus `x1,y1,x2,y2,type` boxes.
Then `build_tiles.py`, `tiling.py`, `tile_dataset.py`, `eval/gate.py`, `training/train.py` (via `--npz`)
and the firmware gate run on PKU with no or minimal changes.

The binarization must be **the same algorithm the ESP32 runs on camera frames**. Then training data,
the printed-photo demo, and the real-board demo all pass through one identical preprocessing step.

## Step 0 — Protect the old method (do first, no approval needed)

- Check for local or unpushed work newer than `origin/main` (last GitHub commit is Nov 3; branch
  `claude/lucid-cannon-eyj5yg` has unmerged Nov 4 results). Report what you find before doing anything else.
- Tag the current state: `git tag deeppcb-v1` (on the newest commit that has the DeepPCB results).
- Do all new work on a branch `pku-dataset`.
- Add a `--dataset {deeppcb,pku}` option where a script needs it. Default stays `deeppcb`.
- Keep DeepPCB outputs in their current paths. Write PKU outputs to `data/tiles_pku/`, `runs_pku/`, etc.

## Step 1 — Inspect PKU and report (no code changes in the pipeline)

Download PKU-Market-PCB. Sources: the official PKU page (robotics.pkusz.edu.cn → Open resources), or a
Kaggle mirror (for example `akhatova/pcb-defects`). Confirm the folder structure: `images/<defect type>/`,
`Annotations/` (VOC XML), `PCB_USED/` (10 templates), `rotation/`. Use the non-rotated images.
Keep `rotation/` only as an optional robustness test later.

Report:
- Number of images, boards, and boxes per class. Image sizes.
- Which template belongs to each test image (from the file name), and the board count per template.
- **px/mm** of the photos. Measure it from a known feature (2.54 mm header pitch, or a board dimension)
  on at least 2 boards. DeepPCB is ~48 px/mm. State the resample factor needed to match the scale our
  DeepPCB pipeline uses (check whether the code uses native 48 px/mm or the 0.5× / 24 px/mm plan).
- Alignment: PKU says test images are in the same position as their template. Measure the residual with an
  x/y shift search that minimizes XOR count (same method as firmware alignment). Report max and median shift.
- Box sizes in px **after resampling**, per class. Compare to DeepPCB (median short side 31 px).
- Class mapping. PKU: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper.
  DeepPCB: open, short, mousebite, spur, spurious copper, pin-hole. Five match. Propose: keep the
  6-output head and use slot 6 = missing_hole in PKU mode. Do not merge missing_hole with pin-hole;
  they are different defects.

**Checkpoint 1: stop and show me this report.**

## Step 2 — Shared binarization (Python ⇄ firmware parity)

- If `pcb_binarize.h` (flat-field, global Otsu, polarity invert) exists locally, port it to
  `data/binarize.py`. If it does not exist, write both versions now.
- Dataset photos have no flat-field frame. Use flat-field = off for dataset images, on for camera frames.
  Everything after flat-field must be identical.
- Channel choice: test gray vs R vs G vs B for copper/laminate contrast under green solder mask.
  Choose one and use the same one in firmware.
- Add a parity test like `tests/test_gate_parity.py`: compile the C version and check exact agreement
  on real PKU crops.

## Step 3 — Converter and the tile check (the important go/no-go step)

`data/pku.py`:
1. Load template + test (color). Resample to the pipeline px/mm.
2. Align test to template (shift search). Binarize both with `data/binarize.py`.
3. Cut aligned 640×640 crops from the pair. Clip boxes to each crop; keep a box only if ≥ 70% of it is
   inside the crop (same rule as tiling). Keep a share of fully clean crops (PKU defects are sparse — this
   is realistic and is where the gate saves energy).
4. Output an index and arrays that `build_tiles.py` accepts. Add `--split-by board`: split train/val/test
   by **template (board)**, never by image or crop. Suggested: test = 2 boards, val = 1–2 boards, train = rest.
   Report which boards are in each fold. The printed demo must use test-fold boards only.

Then run these checks and report the numbers next to the DeepPCB numbers:
- **Defect visibility after binarization, per class:** share of boxes with ≥ 1 XOR pixel left after 3×3
  erosion. If a class drops a lot (copper under solder mask can lose contrast), say so.
- **Tile reachability:** run `data/grid_sweep.py` on PKU. Recall ceiling for the 5×5 / 160 / 120 grid with
  the 70% rule, and whether the 40 px overlap is ≥ nearly all box sizes. If a different tile/stride is needed,
  show the sweep table; do not change the default.
- **Gate:** `eval/gate.py` skip rate vs defect-tile recall on PKU val. Compare with DeepPCB
  (88% skip at 98% recall, radius 1).
- Visual sanity sheet: 10 random crops with test, template, XOR and boxes drawn.

**Checkpoint 2: stop and show me the numbers and the sheet.**

Fallback only if visibility or gate recall is poor: keep the same gate structure but binarize the
**difference** instead of the images: `|test − template| > t` → 3×3 erosion → pixel count. Stage 2 then
gets grayscale tiles. Explain the cost (gate parity test, firmware gate change) before doing it.

## Step 4 — Train with the existing code

- Build `data/tiles_pku/*.npz` with `build_tiles.py --dataset pku --split-by board`.
- Run the same run matrix (`training/run_matrix.py`) with `--npz` pointing at PKU and `runs_pku/` as output:
  teacher, control, Hinton KD, defect-aware KD, gate on/off. Same seeds as DeepPCB.
- Optional, only if time allows: one run trained on DeepPCB + PKU together, evaluated on each test set.

## Step 5 — Camera-domain check and demo assets

- Script `tools/make_print.py`: export a held-out PKU board (template + one defective image) as a PDF at the
  print scale that gives the target px/mm at the current stand height (82 mm lens height). Color, matte.
- Firmware: add a "capture template" button/command to live mode — the camera captures the printed
  template, binarizes it, and stores it as the golden reference. Then the defective print is run through
  align → gate → Stage 2. The DeepPCB benchmark mode stays as it is.
- Report two numbers: PKU test (binarized on PC) vs the same boards printed and captured by the ESP32.
  The difference is the domain gap. This is also an ablation for the paper.

## Step 6 — Documentation for the team

- `Findings.md` and `README.md`: add a "Two datasets" section. DeepPCB results stay as they are,
  marked as `deeppcb-v1`. PKU results go beside them in the same table format.
- Explain in 3–4 lines why we added PKU (domain gap to the live demo) and what did not change.

## Rules

- Small commits, one step each, attributable messages.
- Do not delete DeepPCB code, results, or checkpoints.
- When a number from this brief and a measured number disagree, trust the measured number and tell me.
