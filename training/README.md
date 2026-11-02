# training/

Model definitions, losses and train loops. Nothing here yet.

Planned:

* `models.py` — the ResNet-18 teacher (2-channel input, 6-way multi-label head), the
  depthwise-separable student (~50–150K params), and the micro-CNN learned gate (~5–10K params).
* `losses.py` — the **defect-aware distillation loss**, the project's novel contribution:

  ```
  L = sum_tiles  w(tile) * [ BCE(sigmoid(z_s), y) + lambda * BCE(sigmoid(z_s/T), sigmoid(z_t/T)) ]
  ```

  with `w = 1` for clean tiles and `alpha > 1` for defect tiles, times the inverse
  class-frequency weight. Setting `w = 1` everywhere reduces it to plain Hinton KD, which is the
  prior-method comparison, so both live in one implementation.
  Per-class positive weights come from `TileData.class_pos_weights()`.
* `train.py` — train loops for teacher, control student and KD student, plus QAT.

The loss must honour the per-tile **ignore mask** from the tile dataset: a channel marked
ignored contributes nothing. `TileData` yields `(x, y, ignore)` for exactly this reason.
