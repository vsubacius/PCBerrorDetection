"""Tests for the PC-side Stage-1 gate.

The gate module exists so thresholds can be chosen on validation data and then
used unchanged on the chip, so the properties that matter are agreement with
the firmware's per-tile erosion and honesty about unreachable targets.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))
sys.path.insert(0, str(_REPO / "eval"))

import gate  # noqa: E402
from tile_dataset import TileData, binary_morph  # noqa: E402
from tiling import TileGrid, label_tiles  # noqa: E402

GRID = TileGrid(160, 120)


def make_fold(n=6, seed=0) -> TileData:
    rng = np.random.default_rng(seed)
    test = rng.integers(0, 2, (n, 640, 640)).astype(np.uint8)
    temp = test.copy()
    boxes = []
    for i in range(n):
        x0, y0 = 20 + 30 * i, 20
        # A solid blob: survives a 3x3 erosion, unlike a 1-px fringe.
        test[i, y0:y0 + 30, x0:x0 + 30] = 1
        temp[i, y0:y0 + 30, x0:x0 + 30] = 0
        boxes.append(np.array([[x0, y0, x0 + 30, y0 + 30, 3]], dtype=np.int32))
    ys, igs = zip(*(label_tiles(b, GRID) for b in boxes))
    return TileData(test=test, temp=temp, y=np.stack(ys), ignore=np.stack(igs),
                    boxes=list(boxes), names=np.array([f"i{i}" for i in range(n)]),
                    groups=np.array(["g"] * n), grid=GRID,
                    meta={"pos_thresh": 0.7, "ignore_thresh": 0.0})


def test_scores_are_one_per_tile():
    fold = make_fold()
    scores = gate.board_scores(fold, 0)
    assert scores.shape == (GRID.n_tiles,)
    assert scores.dtype == np.int64


def test_identical_images_score_zero_everywhere():
    test = np.random.default_rng(0).integers(0, 2, (640, 640)).astype(np.uint8)
    assert np.all(gate.tile_scores(test, test, GRID) == 0)


def test_erosion_is_applied_per_tile_matching_the_firmware():
    """Per-tile, not whole-board: that is what gate.cpp can do cheaply."""
    rng = np.random.default_rng(0)
    test = rng.integers(0, 2, (640, 640)).astype(np.uint8)
    temp = rng.integers(0, 2, (640, 640)).astype(np.uint8)
    got = gate.tile_scores(test, temp, GRID, radius=1)
    xor = (test ^ temp).astype(np.uint8)
    expected = np.array([binary_morph(t, False, 1).sum() for t in GRID.cut(xor)])
    assert np.array_equal(got, expected)


def test_radius_zero_is_raw_xor_counting():
    rng = np.random.default_rng(1)
    test = rng.integers(0, 2, (640, 640)).astype(np.uint8)
    temp = rng.integers(0, 2, (640, 640)).astype(np.uint8)
    got = gate.tile_scores(test, temp, GRID, radius=0)
    xor = (test ^ temp).astype(np.uint8)
    assert np.array_equal(got, GRID.cut(xor).reshape(GRID.n_tiles, -1).sum(1))


def test_erosion_reduces_scores_monotonically_in_radius():
    rng = np.random.default_rng(2)
    test = rng.integers(0, 2, (640, 640)).astype(np.uint8)
    temp = rng.integers(0, 2, (640, 640)).astype(np.uint8)
    raw = gate.tile_scores(test, temp, GRID, 0)
    one = gate.tile_scores(test, temp, GRID, 1)
    two = gate.tile_scores(test, temp, GRID, 2)
    assert np.all(one <= raw) and np.all(two <= one)


def test_a_solid_defect_blob_survives_a_3x3_erosion():
    fold = make_fold(n=1)
    assert gate.board_scores(fold, 0, radius=1).max() > 0


def test_clean_mode_uses_the_template_as_the_test_image():
    fold = make_fold()
    assert np.all(gate.board_scores(fold, 0, clean=True) == 0)


def test_fired_matches_the_firmware_convention():
    """gate_fires() in the firmware is strictly `score > threshold`."""
    scores = np.array([0, 1, 5, 6])
    assert list(gate.fired(scores, 5)) == [False, False, False, True]


def test_recall_and_skip_move_in_opposite_directions():
    fold = make_fold()
    scores = gate.fold_scores(fold)
    recalls, skips = [], []
    for threshold in (0, 5, 20, 100, 10_000):
        recall, skip = gate.tile_recall_and_skip(fold, scores, threshold)
        recalls.append(recall)
        skips.append(skip)
    assert recalls == sorted(recalls, reverse=True)
    assert skips == sorted(skips)


def test_chosen_threshold_meets_its_target():
    fold = make_fold(n=8)
    threshold, recall, skip = gate.choose_threshold(fold, 0.98, radius=1)
    assert threshold is not None
    assert recall >= 0.98
    achieved, _ = gate.tile_recall_and_skip(fold, gate.fold_scores(fold, 1), threshold)
    assert achieved >= 0.98


def test_an_unreachable_target_returns_no_threshold():
    """Must not invent an operating point the gate cannot reach."""
    fold = make_fold(n=4)
    scores = gate.fold_scores(fold, radius=1)
    # Force a ceiling below the target by zeroing every defect tile's score.
    positive = [(fold.y[i] > 0).any(axis=1) for i in range(len(fold))]
    for s, p in zip(scores, positive):
        s[p] = 0
    threshold, ceiling, skip = gate.choose_threshold(fold, 0.98, 1, scores)
    assert threshold is None
    assert ceiling < 0.98
    assert np.isnan(skip)


def test_gate_masks_are_usable_as_board_predictions():
    from metrics import BoardPrediction, evaluate_defects

    fold = make_fold(n=5)
    masks = gate.gate_masks(fold, threshold=5, radius=1)
    scores = [np.where(fold.y[i] > 0, 5.0, -5.0) for i in range(len(fold))]
    preds = [BoardPrediction(scores=s, gate_fired=m) for s, m in zip(scores, masks)]
    result = evaluate_defects(
        fold.boxes, preds, fold.grid, 0.0,
        labels_per_board=[(fold.y[i], fold.ignore[i]) for i in range(len(fold))])
    # The blobs are solid, so the gate keeps them and the oracle still finds them.
    assert result.n_found == result.n_defects
    assert result.n_tiles_skipped > 0, "the gate should skip the clean tiles"


def test_the_classical_gate_radius_is_one():
    """Radius 2 caps tile recall near 85%; the firmware hard-codes radius 1."""
    assert gate.DEFAULT_RADIUS == 1
    header = (_REPO / "firmware" / "pcb_bench" / "main" / "gate.hpp").read_text()
    assert "kGateRadius = 1" in header


def test_real_fold_reproduces_the_documented_gate_ordering():
    """3x3 erosion must skip far more clean tiles than raw XOR at equal recall,
    and 5x5 must fail to reach the target at all."""
    npz = _REPO / "data" / "tiles" / "deeppcb_tiles_t160_s120.npz"
    if not npz.exists():
        pytest.skip("tile dataset not built (python data/build_tiles.py)")

    fold = TileData.load(npz, "val")
    for attr in ("test", "temp", "y", "ignore"):
        setattr(fold, attr, getattr(fold, attr)[:30])
    fold.boxes = fold.boxes[:30]

    raw = gate.choose_threshold(fold, 0.98, radius=0)
    eroded = gate.choose_threshold(fold, 0.98, radius=1)
    wide = gate.choose_threshold(fold, 0.98, radius=2)

    assert raw[0] is not None and eroded[0] is not None
    assert eroded[2] > raw[2] + 0.15, "3x3 erosion should skip far more clean tiles"
    assert wide[0] is None, "5x5 erosion cannot reach 98% defect-tile recall"
