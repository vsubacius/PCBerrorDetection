# PCB Defect Detection on ESP32-S3 — Project Context

Team course project for ECE 690OD (On-Device ML, UMass Amherst, Fall 2026). Graded to the 690OD bar:
novel contribution implemented by the team (not library calls), comparison to at least one prior method
from the literature, at least two ablations, on-device measurements (latency, memory, energy) vs. a
sensible baseline, public GitHub repo with docs, 4–6 page workshop-style paper, demo video.

Final deliverables due Dec 9, 2026. Mid-project presentation Nov 4–6 (state the contribution there and get instructor sign-off).

## Hardware
- Target: Seeed XIAO ESP32-S3 Sense (ESP32-S3, 8 MB PSRAM, 8 MB flash, onboard camera, microSD slot, WiFi).
- Training happens in Google Colab (GPU). Code lives in this repo; notebooks are opened in Colab from GitHub.

## Dataset
- **DeepPCB** (github.com/tangsanli5201/DeepPCB): 1,500 pairs of 640×640 binarized, pre-aligned images
  (defect-free template + defective test image). Official split: 1,000 train / 500 test.
- Annotation format per defect: `x1,y1,x2,y2,type` with types 1=open, 2=short, 3=mousebite, 4=spur,
  5=spurious copper, 6=pin-hole.
- Defects were added artificially (~3–12 per image), so they are much denser than on real boards.
- Research use only.
- Previously considered: Roboflow `jr-mqqnk/pcb-defects-detection-anddl` (PKU-Market-PCB style; large color
  images, tiny defects, no clean images). Dropped in favor of DeepPCB.
- A teammate's earlier baseline (different dataset): github.com/BlueDoggoy/Lightweight-PCB-Fault---690D

## Proposed contribution
"A reference-guided, gated tile cascade with defect-aware distillation for bare-PCB inspection on an MCU."

### Pipeline
1. Split each 640×640 image pair into a 5×5 grid of 160×160 tiles, stride 120 (40 px overlap).
   Confirm tile/overlap against the defect-size histogram: overlap should be ≥ nearly all defect sizes.
2. Stage 1 — gate: decides whether a tile differs from its template. Clean tiles exit early.
   - Classical gate: pixel count of XOR(test, template) above a threshold.
   - Learned gate: micro-CNN (~5–10K params) on a downsampled tile.
3. Stage 2 — classifier: multi-label (6 sigmoid outputs) int8 CNN, run only on flagged tiles.
4. Merge overlapping tiles into a list of (defect type, tile location).

### Tile labeling rules
- Tile is positive for a class if it contains ≥ ~70% of that defect box's area.
- Smaller fragments of a defect → that class is "ignore" (masked out of the loss).
- No defects → clean.
- Split train/val **by image, not by tile** (overlapping tiles leak otherwise). Val comes from the official 1,000 train images.

### Augmentation
- Flips and 90° rotations applied identically to test and template.
- Small random shifts (a few px) between test and template to simulate misalignment.
- Random erosion/dilation to simulate binarization-threshold variation.

### Input representations (ablation)
- A: test tile only (1 ch) · B: test + template (2 ch) · C: test + XOR(test, template) (2 ch)

### Models
- Teacher (Colab only): ResNet-18 adapted to 2-channel input, 6-way multi-label head.
- Stage 2 student / control model: team-designed small depthwise-separable CNN, ~50–150K params, int8.
- Gate: classical or learned (see above).

### Defect-aware distillation loss (team-implemented — this is the core novel part)
```
L = Σ_tiles  w(tile) · [ BCE(σ(z_s), y) + λ · BCE(σ(z_s/T), σ(z_t/T)) ]
```
- w = 1 for clean tiles, α > 1 for defect tiles, times inverse class-frequency weight; optionally higher where the teacher is uncertain.
- w = 1 everywhere reduces to plain Hinton-style KD (used as a prior-method comparison).

### Quantization / deployment
- QAT to int8. Toolchain undecided: ESP-DL (ESP-PPQ quantizer) vs. TFLite Micro (esp-nn). Check current docs.
- **First priority: get a dummy model running on the board** to de-risk deployment.
- Optional: structured channel pruning on Stage 2 if over the latency budget.

## Firmware
- Benchmark mode (all reported numbers): load test/template images from microSD into PSRAM, run 25 tiles
  through gate → classifier, log per-stage latency (`esp_timer_get_time()`), memory
  (`heap_caps_get_minimum_free_size()`), and predictions over serial.
- Also run the 500 test-set templates as defect-free "boards" so results cover clean, defective, and mixed sets
  (DeepPCB's defect density is unrealistically high; gate savings show up on mostly-clean data).
- Live demo mode: camera → grayscale → crop → Otsu binarization → align to stored template via small x/y
  shift search minimizing XOR count → same tile pipeline → results served over WiFi. Demo target: a printed
  DeepPCB test image. Live accuracy is not a reported metric.

## Evaluation
- Latency per tile / per stage / per board, gate on vs. off.
- Peak SRAM and PSRAM, model flash size.
- Energy per board via USB power meter or INA219 (avg power × time).
- Accuracy degradation: same test set as PC float vs. PC int8 vs. on-device int8.
- Headline metric: defect-level recall; also precision and false alarms on clean boards.
  A defect counts as found if a tile fully containing it predicts its class above threshold.
- DeepPCB's official metric (IoU > 0.33 boxes) doesn't apply to tile outputs; evaluate all methods with the
  defect-hit metric. A FOMO-style grid head on Stage 2 is a possible later upgrade.

## Rubric mapping
- Naive baseline: control student on all 25 tiles, test-only input, PTQ int8, no gate, no KD.
- Prior methods: Hinton KD (w = 1) and Edge Impulse FOMO on the same device.
- Ablations: input A/B/C; gate none/classical/learned; defect-aware KD vs. plain KD vs. none; QAT vs. PTQ; tile size.
- Key figure: gate threshold sweep — energy per board vs. recall, on clean and defective boards.

## Timeline
- Nov 1–7: tiling + label pipeline; dummy model on ESP32; contribution stated at mid-project presentation.
- Nov 8–14: teacher + control student; QAT; first on-device numbers.
- Nov 15–21: defect-aware KD, both gates, FOMO + Hinton KD comparisons.
- Nov 22–28: ablation training; all on-device measurements (Thanksgiving week).
- Nov 29–Dec 5: live demo mode, energy numbers, paper draft.
- Dec 6–9: video, slides, README.

## Suggested repo layout
```
data/          # download + tiling scripts (no raw data committed)
notebooks/     # Colab notebooks (teacher, student, KD, QAT)
training/      # model defs, losses, train loops
export/        # quantization + conversion to ESP-DL / TFLM
firmware/      # ESP-IDF project for the XIAO ESP32-S3
eval/          # metrics, plots, on-device log parsing
paper/
```

## First task
Write a Colab-ready notebook that downloads DeepPCB, plots the defect-size histogram per class, and builds
the labeled tile dataset (with ignore masks and an image-level train/val split).

## Team notes
Individual contributions are graded separately; keep commits attributable per member.
