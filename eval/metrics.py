"""The one shared metric module. Every run in the experiment matrix uses this.

Defect-level rule
-----------------
A defect counts as **found** if a tile holding at least ``pos_thresh`` (0.70) of
its area predicts that defect's class above threshold. This is deliberately the
*same* rule the tile labels use, so the metric measures the model rather than
the grid.

The older rule in ``CLAUDE.md`` -- a tile *fully containing* the defect -- caps
recall at 95.17% on trainval and 95.89% on test for purely geometric reasons,
so it is not used here. Under the 0.70 rule the ceiling is 99.93% on trainval
and **100.00%** on the official test split: all five unreachable defects
(best-containment 0.669-0.699) sit in trainval, none in test.

Because tiles overlap by 40 px, more than one tile can hold >= 70% of the same
small defect. ``match="any"`` (the default) counts the defect as found if *any*
qualifying tile fires, which is what a detector is actually asked to do.
``match="best"`` restricts it to the single best-containment tile, which is the
stricter reading of the plan's wording. Both are reported; the 0.70 threshold
itself is still pending team confirmation.

Precision and false positives
-----------------------------
Recall is counted over *defects*; precision is counted over *(tile, class)*
firings, because a false positive is not attached to any defect. The two
therefore have different denominators, which is stated in every report this
module prints rather than hidden.

Per the experiment plan:

* a firing on a class the tile has marked **ignore** is neither a true nor a
  false positive -- it is excluded entirely (the tile holds a fragment too
  small to name, so neither answer is wrong);
* a firing on a non-ignored negative class is a false positive.

The gate
--------
A tile the gate skips produces **no** predictions. Any defect it held is lost
and stays lost: ``gate_fired`` masks predictions off, and recall drops
accordingly. That is the whole point of the gate-threshold sweep, so the metric
must never quietly credit a skipped tile.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from deeppcb import CLASS_IDS, CLASS_NAMES, NUM_CLASSES  # noqa: E402
from tiling import POS_THRESH, TileGrid, box_overlap_fracs  # noqa: E402

CLASS_ORDER: tuple[str, ...] = tuple(CLASS_NAMES[c] for c in CLASS_IDS)
MATCH_MODES = ("any", "best")


@dataclass
class BoardPrediction:
    """What the cascade produced for one board.

    Attributes:
        scores: ``(n_tiles, 6)`` per-class scores. Logits by default -- the
            student has no sigmoid -- so thresholds live in logit space too.
        gate_fired: ``(n_tiles,) bool``. False means Stage 2 never ran on that
            tile, so it contributes no predictions at all. Default: all True.
    """

    scores: np.ndarray
    gate_fired: np.ndarray | None = None

    def fired(self, n_tiles: int) -> np.ndarray:
        if self.gate_fired is None:
            return np.ones(n_tiles, dtype=bool)
        return np.asarray(self.gate_fired, dtype=bool)


@dataclass
class DefectResult:
    """Defect-level and tile-level counts, with per-class breakdowns."""

    n_defects: int = 0
    n_found: int = 0
    n_unreachable: int = 0
    n_lost_to_gate: int = 0

    per_class_defects: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    per_class_found: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    per_class_unreachable: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64))

    # (tile, class) firings, for precision.
    true_positives: int = 0
    false_positives: int = 0
    ignored_firings: int = 0
    per_class_tp: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    per_class_fp: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64))

    n_boards: int = 0
    n_tiles: int = 0
    n_tiles_skipped: int = 0

    @property
    def recall(self) -> float:
        """Defect-level recall, with unreachable defects counted as misses."""
        return self.n_found / self.n_defects if self.n_defects else float("nan")

    @property
    def precision(self) -> float:
        """Precision over (tile, class) firings; ignored firings excluded."""
        fired = self.true_positives + self.false_positives
        return self.true_positives / fired if fired else float("nan")

    @property
    def per_class_recall(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.per_class_defects > 0,
                            self.per_class_found / np.maximum(self.per_class_defects, 1),
                            np.nan)

    @property
    def per_class_precision(self) -> np.ndarray:
        fired = self.per_class_tp + self.per_class_fp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(fired > 0, self.per_class_tp / np.maximum(fired, 1), np.nan)

    @property
    def tile_skip_rate(self) -> float:
        return self.n_tiles_skipped / self.n_tiles if self.n_tiles else float("nan")

    def describe(self) -> str:
        lines = [
            f"boards {self.n_boards}   tiles {self.n_tiles}   "
            f"skipped {self.n_tiles_skipped} ({self.tile_skip_rate * 100:.1f}%)",
            f"defects {self.n_defects}   found {self.n_found}   "
            f"recall {self.recall * 100:.2f}%",
        ]
        if self.n_unreachable:
            lines.append(
                f"  of which unreachable by the grid: {self.n_unreachable} "
                f"(counted as misses)")
        if self.n_lost_to_gate:
            lines.append(f"  lost to the gate: {self.n_lost_to_gate}")
        lines.append(
            f"firings: TP {self.true_positives}  FP {self.false_positives}  "
            f"ignored {self.ignored_firings}  precision {self.precision * 100:.2f}%")
        lines.append("")
        lines.append(f"  {'class':<12}{'defects':>9}{'found':>7}{'recall':>9}"
                     f"{'TP':>7}{'FP':>7}{'prec':>9}")
        rec, prec = self.per_class_recall, self.per_class_precision
        for i, name in enumerate(CLASS_ORDER):
            lines.append(
                f"  {name:<12}{self.per_class_defects[i]:>9}"
                f"{self.per_class_found[i]:>7}{rec[i] * 100:>8.2f}%"
                f"{self.per_class_tp[i]:>7}{self.per_class_fp[i]:>7}"
                f"{prec[i] * 100:>8.2f}%")
        lines.append("")
        lines.append("recall counts defects; precision counts (tile, class) firings.")
        return "\n".join(lines)


@dataclass
class CleanBoardResult:
    """False alarms when a defect-free board is inspected.

    Built by running the 500 test templates as boards whose test image *is*
    their template, so every tile is genuinely clean and any firing is a false
    alarm. DeepPCB's defect density is unrealistically high, so this is where
    the gate's value actually shows.
    """

    n_boards: int = 0
    n_tiles: int = 0
    n_tiles_skipped: int = 0
    n_firings: int = 0
    n_boards_with_firing: int = 0
    per_class_firings: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64))

    @property
    def board_false_alarm_rate(self) -> float:
        """Share of clean boards that produce at least one firing."""
        return self.n_boards_with_firing / self.n_boards if self.n_boards else float("nan")

    @property
    def tile_false_alarm_rate(self) -> float:
        return self.n_firings / self.n_tiles if self.n_tiles else float("nan")

    @property
    def tile_skip_rate(self) -> float:
        return self.n_tiles_skipped / self.n_tiles if self.n_tiles else float("nan")

    def describe(self) -> str:
        return (
            f"clean boards {self.n_boards}   tiles {self.n_tiles}   "
            f"skipped {self.n_tiles_skipped} ({self.tile_skip_rate * 100:.1f}%)\n"
            f"false alarms: {self.n_firings} firings on "
            f"{self.n_boards_with_firing} boards "
            f"({self.board_false_alarm_rate * 100:.2f}% of boards, "
            f"{self.tile_false_alarm_rate * 100:.3f}% of tiles)\n"
            + "  ".join(f"{n}={c}" for n, c in
                        zip(CLASS_ORDER, self.per_class_firings))
        )


def _as_thresholds(thresholds) -> np.ndarray:
    """Broadcast a scalar or per-class threshold to ``(6,)``."""
    arr = np.broadcast_to(np.asarray(thresholds, dtype=np.float64),
                          (NUM_CLASSES,)).astype(np.float64)
    return arr


def qualifying_tiles(
    boxes: np.ndarray,
    grid: TileGrid,
    pos_thresh: float = POS_THRESH,
    match: str = "any",
) -> list[np.ndarray]:
    """Tiles that may be credited with finding each box.

    Returns one array of tile indices per box -- empty when the box is
    unreachable, i.e. no tile holds ``pos_thresh`` of its area.
    """
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, got {match!r}")
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return []

    fracs = box_overlap_fracs(boxes, grid.tiles())     # (n_boxes, n_tiles)
    out = []
    for row in fracs:
        ok = np.flatnonzero(row >= pos_thresh)
        if match == "best" and ok.size:
            ok = np.array([int(row.argmax())], dtype=np.int64)
        out.append(ok.astype(np.int64))
    return out


def evaluate_defects(
    boxes_per_board,
    predictions,
    grid: TileGrid,
    thresholds,
    labels_per_board=None,
    pos_thresh: float = POS_THRESH,
    match: str = "any",
) -> DefectResult:
    """Defect-level recall plus (tile, class) precision over a set of boards.

    Args:
        boxes_per_board: iterable of ``(n_i, 5)`` arrays of
            ``[x1, y1, x2, y2, class_id]`` with class ids 1..6.
        predictions: iterable of :class:`BoardPrediction`, aligned with boxes.
        grid: the tile grid the predictions were produced on.
        thresholds: scalar or ``(6,)`` of per-class decision thresholds, in the
            same space as ``scores`` (logits unless stated otherwise).
        labels_per_board: optional iterable of ``(y, ignore)`` pairs from
            ``tiling.label_tiles``. Needed to separate false positives from
            ignored firings; recomputed from the boxes when omitted.
        pos_thresh: area fraction that lets a tile be credited with a defect.
        match: ``"any"`` qualifying tile, or only the ``"best"`` one.

    Returns:
        A :class:`DefectResult`.
    """
    from tiling import label_tiles  # local import: keeps module import cheap

    thr = _as_thresholds(thresholds)
    result = DefectResult()
    n_tiles = grid.n_tiles

    labels_iter = iter(labels_per_board) if labels_per_board is not None else None

    for board_index, (boxes, pred) in enumerate(zip(boxes_per_board, predictions)):
        boxes = np.asarray(boxes)
        if boxes.size == 0:
            boxes = np.zeros((0, 5), dtype=np.int32)

        if labels_iter is not None:
            y, ignore = next(labels_iter)
        else:
            y, ignore = label_tiles(boxes, grid, pos_thresh)

        scores = np.asarray(pred.scores, dtype=np.float64)
        if scores.shape != (n_tiles, NUM_CLASSES):
            raise ValueError(
                f"board {board_index}: scores must be ({n_tiles}, {NUM_CLASSES}), "
                f"got {scores.shape}"
            )
        fired = pred.fired(n_tiles)

        # A skipped tile predicts nothing, whatever its scores say.
        above = (scores > thr[None, :]) & fired[:, None]

        result.n_boards += 1
        result.n_tiles += n_tiles
        result.n_tiles_skipped += int((~fired).sum())

        # --- defect-level recall ---
        tiles_for_box = qualifying_tiles(boxes, grid, pos_thresh, match)
        for box, tiles in zip(boxes, tiles_for_box):
            channel = CLASS_IDS.index(int(box[4]))
            result.n_defects += 1
            result.per_class_defects[channel] += 1
            if tiles.size == 0:
                # Unreachable: no tile can carry its label, so it is a miss by
                # geometry. Reported separately so it is never mistaken for a
                # model failure.
                result.n_unreachable += 1
                result.per_class_unreachable[channel] += 1
                continue
            if above[tiles, channel].any():
                result.n_found += 1
                result.per_class_found[channel] += 1
            elif not fired[tiles].any():
                # Every tile that could have found it was gated away.
                result.n_lost_to_gate += 1

        # --- (tile, class) precision ---
        positive = y > 0
        tp = above & positive
        ignored = above & ignore & ~positive
        fp = above & ~positive & ~ignore
        result.true_positives += int(tp.sum())
        result.false_positives += int(fp.sum())
        result.ignored_firings += int(ignored.sum())
        result.per_class_tp += tp.sum(axis=0).astype(np.int64)
        result.per_class_fp += fp.sum(axis=0).astype(np.int64)

    return result


def evaluate_clean_boards(predictions, grid: TileGrid, thresholds) -> CleanBoardResult:
    """False alarms on defect-free boards.

    Every tile is clean by construction, so there are no ignore masks and every
    firing is a false alarm.
    """
    thr = _as_thresholds(thresholds)
    result = CleanBoardResult()
    n_tiles = grid.n_tiles

    for board_index, pred in enumerate(predictions):
        scores = np.asarray(pred.scores, dtype=np.float64)
        if scores.shape != (n_tiles, NUM_CLASSES):
            raise ValueError(
                f"board {board_index}: scores must be ({n_tiles}, {NUM_CLASSES}), "
                f"got {scores.shape}"
            )
        fired = pred.fired(n_tiles)
        above = (scores > thr[None, :]) & fired[:, None]

        result.n_boards += 1
        result.n_tiles += n_tiles
        result.n_tiles_skipped += int((~fired).sum())
        firings = int(above.sum())
        result.n_firings += firings
        result.n_boards_with_firing += int(firings > 0)
        result.per_class_firings += above.sum(axis=0).astype(np.int64)

    return result


def reachability_ceiling(
    boxes_per_board,
    grid: TileGrid,
    pos_thresh: float = POS_THRESH,
) -> tuple[float, int, int]:
    """``(ceiling, n_reachable, n_total)`` -- the hard upper bound on recall.

    A defect no tile can carry is invisible to the cascade however good Stage 2
    is, so this bounds every recall number reported on the same split.
    """
    total = reachable = 0
    tiles = grid.tiles()
    for boxes in boxes_per_board:
        boxes = np.asarray(boxes)
        if boxes.size == 0:
            continue
        best = box_overlap_fracs(boxes, tiles).max(axis=1)
        total += len(best)
        reachable += int((best >= pos_thresh).sum())
    ceiling = reachable / total if total else float("nan")
    return ceiling, reachable, total


def thresholds_at_tile_rate(
    clean_predictions,
    grid: TileGrid,
    tile_rate: float,
    per_class: bool = True,
) -> np.ndarray:
    """Per-class thresholds at which each class fires on ``tile_rate`` of clean tiles.

    The mechanism behind :func:`choose_thresholds_for_false_alarm_rate`. Each
    class is placed at its *own* score quantile, so a class with a wider score
    distribution gets a correspondingly higher threshold rather than being
    held to another class's scale.
    """
    scores = _stack_live_scores(clean_predictions, grid)
    if per_class:
        return np.array([
            _threshold_for_rate(scores[:, c], tile_rate) for c in range(NUM_CLASSES)
        ])
    shared = _threshold_for_rate(scores.reshape(-1), tile_rate)
    return np.full(NUM_CLASSES, shared)


def _stack_live_scores(clean_predictions, grid: TileGrid) -> np.ndarray:
    """All clean-board scores from tiles the gate actually ran."""
    stacked, fired = [], []
    for pred in clean_predictions:
        stacked.append(np.asarray(pred.scores, dtype=np.float64))
        fired.append(pred.fired(grid.n_tiles))
    if not stacked:
        raise ValueError("no clean-board predictions supplied")
    scores = np.concatenate(stacked)[np.concatenate(fired)]
    if scores.size == 0:
        raise ValueError("the gate skipped every clean tile; cannot set thresholds")
    return scores


def _threshold_for_rate(column: np.ndarray, rate: float) -> float:
    """Lowest threshold with ``(column > t).mean() <= rate``.

    Chosen from the observed values so the realised rate is met from below,
    never exceeded. ``rate <= 0`` returns a threshold nothing can exceed.

    Counted by sorting rather than by comparing every value against every
    candidate: this runs on 500 clean boards for every run and every seed, so
    the O(n^2) form was not affordable.
    """
    if rate <= 0.0:
        return float(np.nextafter(column.max(), np.inf))
    ordered = np.sort(column)
    candidates = np.unique(ordered)
    # Number of values strictly greater than each candidate.
    above = ordered.size - np.searchsorted(ordered, candidates, side="right")
    rates = above / ordered.size
    ok = np.flatnonzero(rates <= rate)      # rates are non-increasing in t
    if ok.size == 0:
        return float(np.nextafter(column.max(), np.inf))
    return float(candidates[ok[0]])


def board_false_alarm_rate(clean_predictions, grid: TileGrid, thresholds) -> float:
    """Share of clean boards producing at least one firing at ``thresholds``."""
    return evaluate_clean_boards(
        clean_predictions, grid, thresholds).board_false_alarm_rate


def choose_thresholds_for_false_alarm_rate(
    clean_predictions,
    grid: TileGrid,
    target_board_rate: float,
    per_class: bool = True,
    n_steps: int = 40,
) -> np.ndarray:
    """Per-class thresholds hitting a target **board-level** false-alarm rate.

    The reported clean-board metric is "share of defect-free boards that raise
    at least one defect", which is what an inspection operator actually
    experiences, so that is the quantity targeted here. A per-class *tile* rate
    is not interpretable as a single number: six classes at 2% of tiles each is
    12% of tiles overall.

    The search sweeps one shared per-class tile rate `q`, places every class at
    its own `q` quantile, and keeps the loosest `q` whose resulting board rate
    still meets the target. Board rate is monotone in `q`, so this is well
    defined.

    Selected on **validation** scores only, per the experiment plan: the
    official test split must never be used to tune anything. The same chosen
    thresholds are then reused unchanged on test.

    Args:
        clean_predictions: :class:`BoardPrediction` for defect-free boards.
        target_board_rate: allowed share of clean boards raising any defect.
        per_class: six thresholds (the frozen choice) or one shared scalar.
        n_steps: bisection iterations on the tile-rate sweep.

    Returns:
        ``(6,)`` thresholds, in the same space as the scores.
    """
    if not 0.0 <= target_board_rate <= 1.0:
        raise ValueError(
            f"target_board_rate must be in [0, 1], got {target_board_rate}")
    boards = list(clean_predictions)

    def thresholds_for(q: float) -> np.ndarray:
        return thresholds_at_tile_rate(boards, grid, q, per_class)

    def meets(thr: np.ndarray) -> bool:
        return board_false_alarm_rate(boards, grid, thr) <= target_board_rate

    strict = thresholds_for(0.0)
    if not meets(strict):
        # Even firing on nothing exceeds the target, which only happens for a
        # negative-ish target; return the strictest thresholds available.
        return strict

    loose = 0.5
    thr_loose = thresholds_for(loose)
    if meets(thr_loose):
        return thr_loose

    # Board rate is monotone in q, so bisect for the loosest q that still fits.
    lo, hi = 0.0, loose                      # lo meets the target, hi does not
    best = strict
    for _ in range(max(1, n_steps)):
        mid = 0.5 * (lo + hi)
        thr = thresholds_for(mid)
        if meets(thr):
            lo, best = mid, thr
        else:
            hi = mid
    return best
