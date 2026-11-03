"""The Stage-1 gate, PC side -- bit-identical to the firmware.

Every gate threshold in this project is chosen here, on validation data, and
then used on the chip. If the two implementations disagree even slightly the
operating point moves and the reported energy/recall trade-off is wrong, so
this module deliberately mirrors ``firmware/pcb_bench/main/gate.cpp``:

* the XOR is eroded **per tile**, not over the whole board, because that is
  what the firmware can do cheaply. Measured difference: eroding tiles
  independently inflates the clean-tile count on 6.8% of clean tiles by a mean
  of 0.4 px, which is negligible -- so no board-wide buffer is needed.
* edge pixels are handled by clamping the sampling index, which is numpy's
  ``pad(mode="edge")`` followed by a windowed minimum.

``tests/test_gate_parity.py`` compiles the C++ and checks the two agree exactly
on random tiles, structured shapes, every radius, and real DeepPCB tiles.

Radius 1 (a 3x3 erosion) is the classical gate. Radius 2 is available for the
ablation but caps defect-tile recall near 85% whatever threshold is used,
because 15.4% of defect tiles score exactly zero under it and a tile scoring
zero cannot be made to fire without also firing on every clean tile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from tile_dataset import TileData, binary_morph  # noqa: E402
from tiling import TileGrid, is_clean  # noqa: E402

#: The classical gate's erosion radius. 1 -> 3x3. See the module docstring
#: before changing it.
DEFAULT_RADIUS = 1

GATE_FEATURES = ("raw", "erode1", "erode2")


def tile_scores(
    test: np.ndarray,
    temp: np.ndarray,
    grid: TileGrid,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    """Eroded-XOR pixel count per tile for one board pair.

    Args:
        test: ``(640, 640)`` binary test image.
        temp: ``(640, 640)`` binary template.
        grid: the tile grid.
        radius: erosion radius; 0 means raw XOR counting.

    Returns:
        ``(n_tiles,) int64`` counts, in tile order.
    """
    xor = (test ^ temp).astype(np.uint8)
    tiles = grid.cut(xor)
    if radius <= 0:
        return tiles.reshape(len(tiles), -1).sum(axis=1).astype(np.int64)
    return np.array(
        [binary_morph(t, dilate=False, radius=radius).sum() for t in tiles],
        dtype=np.int64,
    )


def board_scores(data: TileData, index: int, radius: int = DEFAULT_RADIUS,
                 clean: bool = False) -> np.ndarray:
    """Gate scores for board ``index`` of a fold.

    With ``clean=True`` the template stands in for the test image, so the pair
    matches everywhere -- how a defect-free board is simulated.
    """
    test = data.temp[index] if clean else data.test[index]
    return tile_scores(test, data.temp[index], data.grid, radius)


def fold_scores(data: TileData, radius: int = DEFAULT_RADIUS,
                clean: bool = False) -> list[np.ndarray]:
    """Gate scores for every board in a fold."""
    return [board_scores(data, i, radius, clean) for i in range(len(data))]


def fired(scores: np.ndarray, threshold: int) -> np.ndarray:
    """``score > threshold``, matching ``pcb::gate_fires`` in the firmware."""
    return np.asarray(scores) > threshold


def tile_recall_and_skip(
    data: TileData,
    scores: list[np.ndarray],
    threshold: int,
) -> tuple[float, float]:
    """``(defect-tile recall, clean-tile skip rate)`` at one threshold.

    Fragment-only (ignored) tiles are excluded from both, since neither firing
    nor skipping them is clearly right.
    """
    all_scores = np.concatenate(scores)
    positive, clean_mask = [], []
    for i in range(len(data)):
        y, ignore = data.y[i], data.ignore[i]
        positive.append((y > 0).any(axis=1))
        clean_mask.append(is_clean(y, ignore))
    positive = np.concatenate(positive)
    clean_mask = np.concatenate(clean_mask)

    recall = float((all_scores[positive] > threshold).mean()) if positive.any() else float("nan")
    skip = float((all_scores[clean_mask] <= threshold).mean()) if clean_mask.any() else float("nan")
    return recall, skip


def choose_threshold(
    data: TileData,
    target_tile_recall: float = 0.98,
    radius: int = DEFAULT_RADIUS,
    scores: list[np.ndarray] | None = None,
) -> tuple[int | None, float, float]:
    """Largest threshold on **validation** meeting a defect-tile recall target.

    The largest qualifying threshold is the one that skips the most clean tiles
    while still meeting the target, so it saves the most Stage-2 compute.

    Returns ``(threshold, achieved_recall, clean_skip_rate)``, with
    ``threshold = None`` when the target is unreachable at any threshold >= 0 --
    which is the honest answer for radius 2, not something to paper over.
    """
    scores = scores if scores is not None else fold_scores(data, radius)
    all_scores = np.concatenate(scores)
    positive = np.concatenate([(data.y[i] > 0).any(axis=1) for i in range(len(data))])
    s_pos = all_scores[positive]

    ceiling = float((s_pos > 0).mean())
    if ceiling < target_tile_recall:
        return None, ceiling, float("nan")

    candidates = np.unique(np.concatenate([[0], s_pos[s_pos >= 0]])).astype(np.int64)
    best = None
    for threshold in candidates:
        recall, skip = tile_recall_and_skip(data, scores, int(threshold))
        if recall >= target_tile_recall:
            best = (int(threshold), recall, skip)
        else:
            break                      # recall is non-increasing in threshold
    if best is None:
        return None, ceiling, float("nan")
    return best


def gate_masks(
    data: TileData,
    threshold: int,
    radius: int = DEFAULT_RADIUS,
    clean: bool = False,
) -> list[np.ndarray]:
    """Per-board ``gate_fired`` masks, ready for ``metrics.BoardPrediction``."""
    return [fired(s, threshold) for s in fold_scores(data, radius, clean)]
