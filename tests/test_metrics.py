"""Tests for the shared metric module.

Small, hand-built cases for exactly the situations the experiment plan calls
out: hits, misses, ignored classes, gate skips, and clean-board false alarms.
Every run in the matrix is scored by this code, so a silent change here would
move every reported number at once.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))
sys.path.insert(0, str(_REPO / "eval"))

from deeppcb import CLASS_ID_TO_CHANNEL  # noqa: E402
from metrics import (  # noqa: E402
    BoardPrediction,
    choose_thresholds_for_false_alarm_rate,
    evaluate_clean_boards,
    evaluate_defects,
    qualifying_tiles,
    reachability_ceiling,
    thresholds_at_tile_rate,
)
from tiling import TileGrid  # noqa: E402

GRID = TileGrid(160, 120)
N_TILES = GRID.n_tiles          # 25
N_CLASSES = 6

# A mousebite (class 3 -> channel 2) wholly inside tile 0 and no other tile.
BOX_IN_TILE0 = np.array([[20, 20, 60, 60, 3]])
CH_MOUSEBITE = CLASS_ID_TO_CHANNEL[3]

# A short (class 2 -> channel 1) 240 px wide: no 160 px tile can hold 70% of
# it (best is 160/240 = 66.7%), so it is unreachable by the grid.
BOX_UNREACHABLE = np.array([[200, 20, 440, 40, 2]])
CH_SHORT = CLASS_ID_TO_CHANNEL[2]

# A spur (class 4 -> channel 3) wholly inside the 40 px overlap of tiles 0
# and 1, so *both* tiles hold 100% of it.
BOX_IN_OVERLAP = np.array([[130, 20, 150, 40, 4]])
CH_SPUR = CLASS_ID_TO_CHANNEL[4]


def silent_scores(value: float = -5.0) -> np.ndarray:
    """Scores below any sensible threshold: the model fires nowhere."""
    return np.full((N_TILES, N_CLASSES), value)


def fire(tile: int, channel: int, value: float = 5.0,
         base: float = -5.0) -> np.ndarray:
    scores = np.full((N_TILES, N_CLASSES), base)
    scores[tile, channel] = value
    return scores


# --------------------------------------------------------------------------- #
# hits and misses
# --------------------------------------------------------------------------- #

def test_a_firing_on_the_right_tile_and_class_is_a_hit():
    pred = BoardPrediction(scores=fire(0, CH_MOUSEBITE))
    res = evaluate_defects([BOX_IN_TILE0], [pred], GRID, thresholds=0.0)
    assert res.n_defects == 1 and res.n_found == 1
    assert res.recall == 1.0
    assert res.true_positives == 1 and res.false_positives == 0
    assert res.precision == 1.0
    assert res.per_class_recall[CH_MOUSEBITE] == 1.0


def test_silence_is_a_miss():
    pred = BoardPrediction(scores=silent_scores())
    res = evaluate_defects([BOX_IN_TILE0], [pred], GRID, thresholds=0.0)
    assert res.n_found == 0 and res.recall == 0.0
    assert res.true_positives == 0 and res.false_positives == 0
    assert np.isnan(res.precision), "no firings at all means precision is undefined"


def test_firing_on_the_wrong_class_is_a_miss_and_a_false_positive():
    pred = BoardPrediction(scores=fire(0, CH_SPUR))
    res = evaluate_defects([BOX_IN_TILE0], [pred], GRID, thresholds=0.0)
    assert res.n_found == 0
    assert res.false_positives == 1
    assert res.per_class_fp[CH_SPUR] == 1


def test_firing_on_the_wrong_tile_is_a_miss():
    """Tile 24 is nowhere near the defect, so this is a miss and an FP."""
    pred = BoardPrediction(scores=fire(24, CH_MOUSEBITE))
    res = evaluate_defects([BOX_IN_TILE0], [pred], GRID, thresholds=0.0)
    assert res.n_found == 0
    assert res.false_positives == 1


def test_per_class_thresholds_are_applied_independently():
    scores = np.full((N_TILES, N_CLASSES), -5.0)
    scores[0, CH_MOUSEBITE] = 1.0
    thresholds = np.full(N_CLASSES, 0.0)
    thresholds[CH_MOUSEBITE] = 2.0          # too high for this score
    res = evaluate_defects([BOX_IN_TILE0], [BoardPrediction(scores)], GRID,
                           thresholds=thresholds)
    assert res.n_found == 0
    thresholds[CH_MOUSEBITE] = 0.5          # now low enough
    res = evaluate_defects([BOX_IN_TILE0], [BoardPrediction(scores)], GRID,
                           thresholds=thresholds)
    assert res.n_found == 1


# --------------------------------------------------------------------------- #
# unreachable defects
# --------------------------------------------------------------------------- #

def test_unreachable_defect_is_a_miss_and_reported_separately():
    """No tile holds 70% of it, so no amount of firing can find it."""
    assert qualifying_tiles(BOX_UNREACHABLE, GRID)[0].size == 0
    scores = np.full((N_TILES, N_CLASSES), 5.0)      # fire everywhere
    res = evaluate_defects([BOX_UNREACHABLE], [BoardPrediction(scores)], GRID,
                           thresholds=0.0)
    assert res.n_defects == 1 and res.n_found == 0
    assert res.n_unreachable == 1
    assert res.per_class_unreachable[CH_SHORT] == 1
    assert res.recall == 0.0


def test_reachability_ceiling_bounds_recall():
    ceiling, reachable, total = reachability_ceiling(
        [BOX_IN_TILE0, BOX_UNREACHABLE], GRID)
    assert (reachable, total) == (1, 2)
    assert ceiling == 0.5


# --------------------------------------------------------------------------- #
# ignored classes
# --------------------------------------------------------------------------- #

def test_firing_on_an_ignored_class_is_neither_true_nor_false():
    """Tile 0 holds only half of a box spanning the tile boundary.

    The labelling rule marks that class 'ignore' on tile 0, so a firing there
    is excluded from precision entirely rather than punished.
    """
    boxes = np.array([[150, 20, 170, 40, 2]])        # 50% in tile 0, 100% in tile 1
    pred = BoardPrediction(scores=fire(0, CH_SHORT))
    res = evaluate_defects([boxes], [pred], GRID, thresholds=0.0)
    assert res.ignored_firings == 1
    assert res.false_positives == 0
    assert res.true_positives == 0
    assert res.n_found == 0, "tile 0 does not qualify; only tile 1 does"


def test_the_qualifying_tile_still_counts_the_hit():
    boxes = np.array([[150, 20, 170, 40, 2]])
    pred = BoardPrediction(scores=fire(1, CH_SHORT))
    res = evaluate_defects([boxes], [pred], GRID, thresholds=0.0)
    assert res.n_found == 1 and res.true_positives == 1
    assert res.false_positives == 0 and res.ignored_firings == 0


# --------------------------------------------------------------------------- #
# overlapping tiles: match="any" vs match="best"
# --------------------------------------------------------------------------- #

def test_two_tiles_can_both_qualify_for_one_small_defect():
    tiles = qualifying_tiles(BOX_IN_OVERLAP, GRID, match="any")[0]
    assert set(tiles.tolist()) == {0, 1}
    best = qualifying_tiles(BOX_IN_OVERLAP, GRID, match="best")[0]
    assert best.size == 1


def test_match_any_credits_either_tile_and_match_best_only_one():
    # Fire on tile 1 only. Both tiles hold 100% of the defect.
    pred = BoardPrediction(scores=fire(1, CH_SPUR))
    any_res = evaluate_defects([BOX_IN_OVERLAP], [pred], GRID, thresholds=0.0,
                               match="any")
    assert any_res.n_found == 1

    best_tiles = qualifying_tiles(BOX_IN_OVERLAP, GRID, match="best")[0]
    other = 1 if best_tiles[0] == 0 else 0
    best_res = evaluate_defects([BOX_IN_OVERLAP],
                                [BoardPrediction(scores=fire(other, CH_SPUR))],
                                GRID, thresholds=0.0, match="best")
    assert best_res.n_found == 0, "match='best' must ignore the other tile"


def test_rejects_an_unknown_match_mode():
    with pytest.raises(ValueError):
        qualifying_tiles(BOX_IN_TILE0, GRID, match="nearest")


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #

def test_a_gated_tile_produces_no_predictions_and_the_defect_stays_lost():
    """The gate cannot be allowed to improve recall by omission."""
    scores = fire(0, CH_MOUSEBITE)
    gate = np.ones(N_TILES, dtype=bool)
    gate[0] = False                                  # skip the only tile that matters
    res = evaluate_defects([BOX_IN_TILE0],
                           [BoardPrediction(scores=scores, gate_fired=gate)],
                           GRID, thresholds=0.0)
    assert res.n_found == 0, "a skipped tile must not be credited"
    assert res.n_lost_to_gate == 1
    assert res.true_positives == 0
    assert res.n_tiles_skipped == 1
    assert res.tile_skip_rate == pytest.approx(1 / N_TILES)


def test_the_gate_also_suppresses_false_positives():
    scores = fire(24, CH_MOUSEBITE)
    gate = np.ones(N_TILES, dtype=bool)
    gate[24] = False
    res = evaluate_defects([BOX_IN_TILE0],
                           [BoardPrediction(scores=scores, gate_fired=gate)],
                           GRID, thresholds=0.0)
    assert res.false_positives == 0


def test_gate_off_by_default():
    res = evaluate_defects([BOX_IN_TILE0], [BoardPrediction(fire(0, CH_MOUSEBITE))],
                           GRID, thresholds=0.0)
    assert res.n_tiles_skipped == 0 and res.n_found == 1


def test_lost_to_gate_is_not_counted_when_another_tile_could_still_fire():
    """Only a defect whose every qualifying tile was skipped is 'lost'."""
    scores = fire(1, CH_SPUR)
    gate = np.ones(N_TILES, dtype=bool)
    gate[0] = False                      # tile 0 skipped, tile 1 still runs
    res = evaluate_defects([BOX_IN_OVERLAP],
                           [BoardPrediction(scores=scores, gate_fired=gate)],
                           GRID, thresholds=0.0, match="any")
    assert res.n_found == 1 and res.n_lost_to_gate == 0


# --------------------------------------------------------------------------- #
# clean boards
# --------------------------------------------------------------------------- #

def test_clean_board_firings_are_all_false_alarms():
    boards = [
        BoardPrediction(scores=silent_scores()),
        BoardPrediction(scores=fire(3, CH_SHORT)),
        BoardPrediction(scores=fire(7, CH_SPUR)),
    ]
    res = evaluate_clean_boards(boards, GRID, thresholds=0.0)
    assert res.n_boards == 3
    assert res.n_firings == 2
    assert res.n_boards_with_firing == 2
    assert res.board_false_alarm_rate == pytest.approx(2 / 3)
    assert res.tile_false_alarm_rate == pytest.approx(2 / (3 * N_TILES))
    assert res.per_class_firings[CH_SHORT] == 1


def test_the_gate_reduces_clean_board_false_alarms():
    scores = fire(3, CH_SHORT)
    gate = np.ones(N_TILES, dtype=bool)
    gate[3] = False
    res = evaluate_clean_boards(
        [BoardPrediction(scores=scores, gate_fired=gate)], GRID, thresholds=0.0)
    assert res.n_firings == 0
    assert res.board_false_alarm_rate == 0.0
    assert res.tile_skip_rate == pytest.approx(1 / N_TILES)


# --------------------------------------------------------------------------- #
# threshold selection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", [0.0, 0.05, 0.2, 0.5])
def test_thresholds_meet_the_target_board_rate_from_below(target):
    rng = np.random.default_rng(0)
    boards = [BoardPrediction(scores=rng.normal(size=(N_TILES, N_CLASSES)))
              for _ in range(60)]
    thr = choose_thresholds_for_false_alarm_rate(boards, GRID, target)
    assert thr.shape == (N_CLASSES,)
    res = evaluate_clean_boards(boards, GRID, thresholds=thr)
    assert res.board_false_alarm_rate <= target + 1e-12


def test_a_looser_target_never_gives_stricter_thresholds():
    """Board false-alarm rate is monotone in the threshold sweep."""
    rng = np.random.default_rng(3)
    boards = [BoardPrediction(scores=rng.normal(size=(N_TILES, N_CLASSES)))
              for _ in range(40)]
    rates = [
        evaluate_clean_boards(
            boards, GRID,
            thresholds=choose_thresholds_for_false_alarm_rate(boards, GRID, t),
        ).board_false_alarm_rate
        for t in (0.02, 0.1, 0.3)
    ]
    assert rates == sorted(rates)


def test_per_class_thresholds_track_each_class_own_scale():
    """A class with a wider score spread must get a higher threshold."""
    rng = np.random.default_rng(7)
    scores = np.stack([
        rng.normal(0.0, 1.0 + 2.0 * c, size=(N_TILES,)) for c in range(N_CLASSES)
    ], axis=1)
    boards = [BoardPrediction(scores=scores)]
    thr = thresholds_at_tile_rate(boards, GRID, 0.05)
    assert thr[0] < thr[-1]


def test_shared_threshold_mode_returns_one_value_for_every_class():
    rng = np.random.default_rng(1)
    boards = [BoardPrediction(scores=rng.normal(size=(N_TILES, N_CLASSES)))
              for _ in range(20)]
    thr = choose_thresholds_for_false_alarm_rate(boards, GRID, 0.05, per_class=False)
    assert len(set(thr.tolist())) == 1


def test_rejects_an_impossible_target_rate():
    boards = [BoardPrediction(scores=np.zeros((N_TILES, N_CLASSES)))]
    with pytest.raises(ValueError, match="target_board_rate"):
        choose_thresholds_for_false_alarm_rate(boards, GRID, 1.5)


def test_threshold_selection_ignores_gated_tiles():
    """A tile the gate skips cannot false-alarm, so it must not set the bar."""
    scores = np.full((N_TILES, N_CLASSES), -5.0)
    scores[0, :] = 50.0                       # a huge score on a skipped tile
    gate = np.ones(N_TILES, dtype=bool)
    gate[0] = False
    thr = thresholds_at_tile_rate(
        [BoardPrediction(scores=scores, gate_fired=gate)], GRID, 0.5)
    assert np.all(thr < 50.0)


# --------------------------------------------------------------------------- #
# shape validation and multi-board accumulation
# --------------------------------------------------------------------------- #

def test_wrong_score_shape_is_rejected():
    bad = BoardPrediction(scores=np.zeros((N_TILES, 3)))
    with pytest.raises(ValueError, match="scores must be"):
        evaluate_defects([BOX_IN_TILE0], [bad], GRID, thresholds=0.0)


def test_counts_accumulate_across_boards():
    boards = [BOX_IN_TILE0, BOX_IN_TILE0, BOX_IN_TILE0]
    preds = [
        BoardPrediction(fire(0, CH_MOUSEBITE)),
        BoardPrediction(fire(0, CH_MOUSEBITE)),
        BoardPrediction(silent_scores()),
    ]
    res = evaluate_defects(boards, preds, GRID, thresholds=0.0)
    assert res.n_boards == 3 and res.n_defects == 3 and res.n_found == 2
    assert res.recall == pytest.approx(2 / 3)
    assert res.n_tiles == 3 * N_TILES


def test_board_with_no_defects_contributes_only_false_positives():
    res = evaluate_defects([np.zeros((0, 5), dtype=np.int32)],
                           [BoardPrediction(fire(5, CH_SPUR))], GRID, thresholds=0.0)
    assert res.n_defects == 0 and np.isnan(res.recall)
    assert res.false_positives == 1


def test_describe_runs_and_states_the_mixed_denominator():
    res = evaluate_defects([BOX_IN_TILE0], [BoardPrediction(fire(0, CH_MOUSEBITE))],
                           GRID, thresholds=0.0)
    text = res.describe()
    assert "recall counts defects" in text
    assert "mousebite" in text


# --------------------------------------------------------------------------- #
# integration with the real dataset
# --------------------------------------------------------------------------- #

def test_real_split_ceilings_match_the_documented_values():
    """The 0.70 rule must leave the official test split with a 100% ceiling."""
    import deeppcb

    root = _REPO / "data" / "DeepPCB"
    if not (root / "PCBData").is_dir():
        pytest.skip("DeepPCB not downloaded (python data/build_tiles.py --download)")

    for split, expected_unreachable in (("trainval", 5), ("test", 0)):
        samples = deeppcb.index(root, split)
        ceiling, reachable, total = reachability_ceiling(
            (s.boxes() for s in samples), GRID)
        assert total - reachable == expected_unreachable
        if split == "test":
            assert ceiling == 1.0
        else:
            assert ceiling == pytest.approx(0.9993, abs=1e-4)


# --------------------------------------------------------------------------- #
# average precision (checkpoint-selection metric)
# --------------------------------------------------------------------------- #

def test_average_precision_is_one_for_a_perfect_ranking():
    from metrics import tile_average_precision

    targets = np.zeros((10, N_CLASSES))
    targets[:3, 0] = 1.0
    scores = np.zeros((10, N_CLASSES))
    scores[:3, 0] = 10.0                      # the positives rank first
    per_class, macro = tile_average_precision(targets * 0 + scores, targets,
                                              np.zeros_like(targets, bool))
    assert per_class[0] == pytest.approx(1.0)
    assert macro == pytest.approx(1.0), "classes without positives must not count"


def test_average_precision_is_low_for_an_inverted_ranking():
    from metrics import tile_average_precision

    targets = np.zeros((10, N_CLASSES))
    targets[:3, 0] = 1.0
    scores = np.zeros((10, N_CLASSES))
    scores[:3, 0] = -10.0                     # the positives rank last
    per_class, _ = tile_average_precision(scores, targets,
                                          np.zeros_like(targets, bool))
    assert per_class[0] < 0.5


def test_average_precision_excludes_ignored_entries():
    from metrics import tile_average_precision

    targets = np.zeros((6, N_CLASSES))
    targets[:2, 0] = 1.0
    scores = np.zeros((6, N_CLASSES))
    scores[:2, 0] = 5.0
    scores[2:4, 0] = 50.0                     # would rank above the positives
    ignore = np.zeros_like(targets, bool)
    ignore[2:4, 0] = True                     # but they are ignored
    per_class, _ = tile_average_precision(scores, targets, ignore)
    assert per_class[0] == pytest.approx(1.0)


def test_average_precision_is_nan_for_a_class_with_no_positives():
    from metrics import tile_average_precision

    targets = np.zeros((5, N_CLASSES))
    per_class, macro = tile_average_precision(
        np.zeros((5, N_CLASSES)), targets, np.zeros_like(targets, bool))
    assert np.all(np.isnan(per_class))
    assert np.isnan(macro)


def test_average_precision_beats_chance_on_a_noisy_but_informative_score():
    from metrics import tile_average_precision

    rng = np.random.default_rng(0)
    targets = (rng.random((2000, N_CLASSES)) < 0.1).astype(float)
    scores = targets + rng.normal(0, 0.7, targets.shape)
    _, macro = tile_average_precision(scores, targets,
                                      np.zeros_like(targets, bool))
    assert 0.2 < macro < 1.0
