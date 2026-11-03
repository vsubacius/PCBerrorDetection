"""Tests for the Stage-1 gate threshold reporting.

Regression tests for the bug that produced a wrong claim in Findings.md: the
script reported a threshold derived from a quantile of the defect-tile scores
without ever checking the recall that threshold achieves. When the quantile
lands on a tie mass at zero -- exactly what a 5x5 erosion produces -- the
reported pair is not an operating point the gate can reach.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))
sys.path.insert(0, str(_REPO / "eval"))

from gate_analysis import TARGET_RECALL, analyse, auc, threshold_for_recall  # noqa: E402


def test_unreachable_target_returns_no_threshold():
    """15% of defect tiles at exactly zero caps recall at 85%.

    No threshold >= 0 can fire on a tile scoring 0, and a negative threshold
    fires on every clean tile too, so those defects are unrecoverable. The
    function must say so instead of inventing a threshold.
    """
    s_pos = np.concatenate([np.zeros(150), np.arange(1, 851, dtype=float)])
    thr, achieved = threshold_for_recall(s_pos, 0.98)
    assert thr is None
    assert achieved == pytest.approx(0.85)


def test_reachable_target_gives_an_honest_threshold():
    s_pos = np.concatenate([np.zeros(150), np.arange(1, 851, dtype=float)])
    thr, achieved = threshold_for_recall(s_pos, 0.80)
    assert thr is not None
    assert achieved >= 0.80
    assert (s_pos > thr).mean() == pytest.approx(achieved)


def test_threshold_with_no_zero_scoring_defect_tiles():
    s_pos = np.arange(1, 1001, dtype=float)
    thr, achieved = threshold_for_recall(s_pos, TARGET_RECALL)
    assert thr is not None and achieved >= TARGET_RECALL
    assert (s_pos > thr).mean() == pytest.approx(achieved)


def test_the_returned_threshold_is_the_most_aggressive_one_that_works():
    """A smaller threshold would skip fewer clean tiles for no recall gain."""
    s_pos = np.arange(1, 101, dtype=float)
    thr, achieved = threshold_for_recall(s_pos, 0.90)
    assert achieved >= 0.90
    assert (s_pos > thr + 1).mean() < 0.90, "a larger threshold must break the target"


def test_recall_is_monotone_in_the_threshold():
    rng = np.random.default_rng(0)
    s_pos = rng.integers(0, 500, 2000).astype(float)
    recalls = [(s_pos > t).mean() for t in range(0, 500, 25)]
    assert recalls == sorted(recalls, reverse=True)


@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 0.98, 1.0])
def test_achieved_recall_always_meets_the_target_when_one_is_returned(target):
    rng = np.random.default_rng(1)
    s_pos = np.concatenate([np.zeros(40), rng.integers(1, 300, 960).astype(float)])
    thr, achieved = threshold_for_recall(s_pos, target)
    if thr is not None:
        assert achieved >= target - 1e-12
        assert (s_pos > thr).mean() >= target - 1e-12
    else:
        assert float((s_pos > 0).mean()) < target


def test_auc_handles_ties_at_zero():
    """After erosion most clean tiles score exactly 0, so a tie-naive AUC
    would be noticeably optimistic."""
    s_pos = np.array([0.0, 0.0, 1.0, 2.0])
    s_neg = np.array([0.0, 0.0, 0.0, 0.0])
    value = auc(s_pos, s_neg)
    assert 0.5 < value < 1.0, "ties must be averaged, not counted as wins"
    assert auc(np.array([1.0, 2.0]), np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert auc(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == pytest.approx(0.5)


def test_analyse_reports_the_ceiling_and_the_achieved_recall_on_real_data():
    """The erode2 row must now be internally consistent."""
    from tile_dataset import TileData

    npz = _REPO / "data" / "tiles" / "deeppcb_tiles_t160_s120.npz"
    if not npz.exists():
        pytest.skip("tile dataset not built (python data/build_tiles.py)")

    data = TileData.load(npz, "val")
    # A small slice keeps this fast; the structure of the result is what matters.
    data.test, data.temp = data.test[:25], data.temp[:25]
    data.y, data.ignore = data.y[:25], data.ignore[:25]
    rows = analyse(data, features=("erode1", "erode2"), shifts=(0,))

    for row in rows:
        assert 0.0 <= row["ceiling"] <= 1.0
        assert row["ceiling"] == pytest.approx(1.0 - row["zero_pos"])
        if row["thr"] is None:
            assert row["ceiling"] < TARGET_RECALL
        else:
            assert row["recall_at_thr"] >= TARGET_RECALL - 1e-12

    by_feature = {r["feature"]: r for r in rows}
    # The documented reason radius 1 is the right classical gate: a 3x3
    # erosion leaves no defect tile at zero, a 5x5 erosion deletes many.
    assert by_feature["erode1"]["zero_pos"] < by_feature["erode2"]["zero_pos"]
