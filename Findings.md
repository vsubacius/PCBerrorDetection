## Finding 1: your tile grid works, but not for the reason the plan gives

**The setup.** The board image is 640×640, too big for the chip to chew at once, so you cut it into 25 squares of 160×160 that overlap each other by 40 px. The overlap exists because a defect sitting right on a cut line gets sliced in half, and neither half looks like a complete defect.

**What the plan assumed.** "Overlap should be ≥ nearly all defect sizes" — i.e. 40 px of overlap only works if defects are smaller than 40 px.

**What I measured.** Across all 10,013 defects: median 38 px, but the 90th percentile is 58 px and the biggest is 173 px. So that assumption is false. Taken literally, you'd need ~100+ px of overlap, which means far more tiles and far more compute.

**Why you're fine anyway.** Your labelling rule doesn't require a defect to be *entirely* inside a tile — it says a tile is positive if it holds **≥70% of the defect's area**. A defect only needs *most* of itself in one place. Under that rule only **5 defects out of 10,013** fail to land in any tile. So keep 160/120; just fix the reasoning in the write-up.

**Why this mattered enough to check.** If a defect never reaches 70% in any tile, then no tile ever carries its label — so the model can't find it, no matter how well you train. That's a hard ceiling. It happens to be 99.95%, which is a non-issue, but it was worth confirming rather than assuming, because the fix (more overlap) would have been expensive.

**The one real problem.** Your *evaluation* rule is stricter than your *training* rule. It counts a defect as found only if a tile **fully containing** it fires — and only 95.4% of defects are fully inside any tile. So a flawless model would still score 95.4% recall, and the missing 4.6% is pure geometry, nothing to do with your model. Score against the same ≥70% tile you trained on and the ceiling goes back to 99.95%. **This is a one-line decision and it's worth making before you report any recall number.**

## Finding 2: the gate needs one extra step, and it's a big deal

**The setup.** 25 tiles per board, and running the classifier on all 25 is basically the whole energy budget. But 53% of tiles are clean. So Stage 1 asks, very cheaply, "does this tile differ from its template?" and skips the clean ones. The cheapest possible version: XOR the test tile against the template (giving a 1 wherever they disagree), count the 1s, and fire if the count is high.

**What I found: it barely works.** A *clean* tile already carries ~165 disagreeing pixels. The reason is a dataset artifact — DeepPCB's images are pure black-and-white but stored as JPEG, which smears edges, so re-thresholding leaves a 1-pixel-wide halo of disagreement along *every* copper trace. A PCB has an enormous amount of trace edge, so every tile has a big baseline noise count that drowns out small defects.

In numbers: AUC 0.78 (0.5 is a coin flip, 1.0 is perfect). At a threshold that keeps 98% of defective tiles, you skip only 44% of the clean ones — so the gate saves much less than the plan assumes.

**The fix is one cheap operation.** Erode the XOR mask first: keep a disagreement pixel only if its neighbours also disagree. A 1-px halo has no thick core and vanishes; a real defect is a blob and survives.

| gate | AUC | clean tiles skipped (at 98% defect recall) |
| --- | --- | --- |
| raw XOR count | 0.78 | 44% |
| 3×3 erosion first | **0.95** | **88%** |
| 5×5 erosion first | 0.92 | 98% (and still 90% with ±3 px misalignment) |

An erosion is just an AND over a small window — no multiplies, cheap on the ESP32-S3.

**Two consequences.** Your classical gate should count *eroded* XOR, not raw XOR. And your learned micro-CNN gate now has to beat a baseline that already skips ~90% of clean tiles, rather than a weak one. That's a better paper either way, but it changes what the learned gate has to prove.

## Next steps

Both findings above are good material for it: they show you measured your own design assumptions instead of accepting them, and the gate result sharpens what your contribution actually has to beat.

In priority order:

1. **Dummy model on the board.** Your own plan calls this first priority, and it's the one thing a notebook can't de-risk — the ESP-DL vs. TFLite Micro choice is still open, and toolchain surprises eat days. This is where I'd go next.
2. **Settle the metric** (≥70% tile instead of full containment) and the gate definition (eroded XOR). Both are small edits that affect every number you report afterwards, so cheaper now than later.
3. **Teacher + control student + the defect-aware KD loss.** The loss is your actual novel contribution, and `training/README.md` has the formula and the interfaces sketched out — including the detail that it must honour the per-tile ignore mask.

