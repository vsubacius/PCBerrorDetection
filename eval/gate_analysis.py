"""How good is the classical Stage-1 gate, and what does it depend on?

Stage 1 must drop clean tiles without dropping defective ones. The obvious
classical gate is ``count(XOR(test, template)) > threshold``, and this script
measures how well that actually separates clean from defective tiles on the
built tile dataset -- plus two things that turn out to matter a lot:

**Morphological denoising.** DeepPCB's images are binarized but stored as JPEG,
so re-thresholding leaves a 1-px fringe of disagreement along every copper
edge. The median *clean* tile already carries ~165 XOR pixels, which swamps
small defects. Eroding the XOR mask before counting removes that fringe and
transforms the gate (see ``--features``).

**Misalignment.** The pairs ship pre-registered. A real camera path leaves a
residual shift, so the script re-runs every feature with the template jittered
by +-N px to show how fast the separation decays.

Reported per (feature, shift):

* ``AUC``           -- ranking quality of the feature as a defect score
* ``ceiling``       -- the highest defect-tile recall *any* usable threshold can
                       reach, i.e. the share of defect tiles scoring above zero.
                       A tile scoring exactly 0 can only be made to fire by a
                       negative threshold, which fires on every clean tile too,
                       so such tiles are unrecoverable.
* ``thr``           -- smallest threshold reaching the target recall, or ``n/a``
                       when the ceiling is below it
* ``recall@thr``    -- the recall actually achieved there
* ``skip@thr``      -- share of *clean* tiles skipped at that threshold; this is
                       the Stage-2 compute the cascade actually saves
* ``recall@skip50`` -- defect-tile recall if the gate is tuned to skip half the
                       clean tiles instead

.. warning::

   An earlier version of this script reported ``thr@98``/``skip@98`` as a
   quantile of the defect-tile scores without ever checking the recall that
   threshold achieves. When the 2% quantile lands on a tie mass at zero -- which
   is exactly what happens for a 5x5 erosion -- the reported pair is not an
   operating point the gate can reach. That is why ``Findings.md`` claimed a
   5x5 erosion skips 98% of clean tiles at 98% defect recall: in reality
   ``skip@98 = 98.4%`` and ``recall = 86.9%`` described the *same* threshold of
   zero. The ``ceiling`` and ``recall@thr`` columns exist so that cannot recur.

Run from the repo root::

    python eval/gate_analysis.py --npz data/tiles/deeppcb_tiles_t160_s120.npz

Note the gate is scored on *tiles*, not defects: a tile counts as positive if
any class is positive. Fragment-only (ignored) tiles are excluded from both
groups, since neither firing nor skipping them is clearly right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from tile_dataset import TileData, binary_morph, shift_image  # noqa: E402
from tiling import is_clean  # noqa: E402

TARGET_RECALL = 0.98
TARGET_SKIP = 0.50

#: Gate features, each mapping a 640x640 XOR mask to a 640x640 mask to count.
#: All are cheap on an MCU: a kxk binary erosion is a sliding AND, separable
#: into two 1-D passes, and needs no multiplies.
FEATURES: dict[str, str] = {
    "raw": "count XOR pixels directly",
    "erode1": "3x3 erosion first (drops the 1-px binarization fringe)",
    "erode2": "5x5 erosion first (drops wider fringes; survives misalignment)",
}


def gate_feature(xor: np.ndarray, feature: str) -> np.ndarray:
    if feature == "raw":
        return xor
    if feature == "erode1":
        return binary_morph(xor, dilate=False, radius=1)
    if feature == "erode2":
        return binary_morph(xor, dilate=False, radius=2)
    raise ValueError(f"unknown feature {feature!r}; choose from {sorted(FEATURES)}")


def tile_scores(data: TileData, feature: str, shift: int, seed: int = 0):
    """Per-tile gate score, plus the defect and clean tile masks."""
    rng = np.random.default_rng(seed)
    scores, pos, clean = [], [], []
    for i in range(len(data)):
        temp = data.temp[i]
        if shift:
            dx = int(rng.integers(-shift, shift + 1))
            dy = int(rng.integers(-shift, shift + 1))
            temp = shift_image(temp, dx, dy)
        mask = gate_feature((data.test[i] ^ temp).astype(np.uint8), feature)
        tiles = data.grid.cut(mask)
        scores.append(tiles.reshape(len(tiles), -1).sum(axis=1))
        y, ign = data.y[i], data.ignore[i]
        pos.append((y > 0).any(axis=1))
        clean.append(is_clean(y, ign))
    return (np.concatenate(scores).astype(np.float64),
            np.concatenate(pos), np.concatenate(clean))


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """ROC AUC via rank statistics, with ties averaged.

    Ties matter here: after erosion most clean tiles score exactly 0, so a
    tie-naive AUC would be noticeably optimistic.
    """
    allv = np.concatenate([scores_pos, scores_neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n_p, n_n = len(scores_pos), len(scores_neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def threshold_for_recall(s_pos: np.ndarray, target: float) -> tuple[float | None, float]:
    """Smallest threshold >= 0 whose defect-tile recall reaches ``target``.

    The gate fires on ``score > thr``, so recall is non-increasing in ``thr``
    and the *smallest* qualifying threshold is the one that skips the most
    clean tiles while still meeting the target. Returns ``(None, ceiling)``
    when even ``thr = 0`` cannot reach the target, which happens whenever more
    than ``1 - target`` of defect tiles score exactly zero.
    """
    ordered = np.sort(s_pos)
    ceiling = float((s_pos > 0).mean())
    if ceiling < target:
        return None, ceiling
    candidates = np.unique(np.concatenate([[0.0], ordered[ordered >= 0]]))
    recalls = 1.0 - np.searchsorted(ordered, candidates, side="right") / len(ordered)
    ok = np.flatnonzero(recalls >= target)
    if ok.size == 0:
        return None, ceiling
    best = ok[-1]          # largest threshold still meeting the target
    return float(candidates[best]), float(recalls[best])


def analyse(data: TileData, features=("raw", "erode1", "erode2"),
            shifts=(0, 1, 2, 3), seed: int = 0) -> list[dict]:
    rows = []
    for feature in features:
        for shift in shifts:
            scores, pos, clean = tile_scores(data, feature, shift, seed)
            s_pos, s_neg = scores[pos], scores[clean]
            thr, achieved = threshold_for_recall(s_pos, TARGET_RECALL)
            thr2 = float(np.quantile(s_neg, TARGET_SKIP))
            rows.append({
                "feature": feature, "shift": shift,
                "n_pos": len(s_pos), "n_clean": len(s_neg),
                "median_pos": float(np.median(s_pos)),
                "median_clean": float(np.median(s_neg)),
                "zero_clean": float((s_neg == 0).mean()),
                "zero_pos": float((s_pos == 0).mean()),
                "ceiling": float((s_pos > 0).mean()),
                "auc": auc(s_pos, s_neg),
                "thr": thr,
                "recall_at_thr": achieved,
                "skip": (float((s_neg <= thr).mean()) if thr is not None
                         else float("nan")),
                "recall_skip50": float((s_pos > thr2).mean()),
            })
    return rows


def format_table(rows: list[dict]) -> str:
    head = (f"{'feature':>8}{'shift':>7}{'med clean':>11}{'med defect':>12}"
            f"{'AUC':>8}{'ceiling':>9}{'thr':>6}{'recall@thr':>12}"
            f"{'skip@thr':>10}{'recall@skip50':>15}")
    out = [head, "-" * len(head)]
    last = None
    for r in rows:
        if last is not None and r["feature"] != last:
            out.append("")
        last = r["feature"]
        unreachable = r["thr"] is None
        thr_text = "n/a" if unreachable else f"{r['thr']:.0f}"
        recall_text = ("--" if unreachable
                       else f"{r['recall_at_thr'] * 100:.1f}%")
        skip_text = "--" if unreachable else f"{r['skip'] * 100:.1f}%"
        out.append(
            f"{r['feature']:>8}{str(r['shift']) + 'px':>7}{r['median_clean']:>11.0f}"
            f"{r['median_pos']:>12.0f}{r['auc']:>8.4f}"
            f"{r['ceiling'] * 100:>8.1f}%{thr_text:>6}{recall_text:>12}"
            f"{skip_text:>10}{r['recall_skip50'] * 100:>14.1f}%"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="data/tiles/deeppcb_tiles_t160_s120.npz")
    ap.add_argument("--fold", default="val")
    ap.add_argument("--features", nargs="*", default=list(FEATURES),
                    choices=list(FEATURES))
    ap.add_argument("--shifts", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = TileData.load(args.npz, args.fold)
    print(f"fold={args.fold}  {len(data)} images  {data.grid.describe()}")
    rows = analyse(data, tuple(args.features), tuple(args.shifts), args.seed)
    print(f"{rows[0]['n_pos']} defect tiles vs {rows[0]['n_clean']} clean tiles "
          f"(fragment-only tiles excluded)\n")
    for name in args.features:
        print(f"  {name:<8}{FEATURES[name]}")
    print()
    print(format_table(rows))
    print(
        f"\nceiling         = highest defect-tile recall any threshold >= 0 can reach"
        f"\n                  (defect tiles scoring exactly 0 are unrecoverable)"
        f"\nthr             = smallest threshold reaching {TARGET_RECALL:.0%} recall, or n/a"
        f"\nrecall@thr      = recall actually achieved there -- check this, do not assume it"
        f"\nskip@thr        = share of clean tiles skipped there (= Stage-2 compute saved)"
        f"\nrecall@skip{TARGET_SKIP:.0%}   = defect-tile recall when tuned to skip {TARGET_SKIP:.0%} of clean tiles"
        "\nshift           = template jittered up to +-N px per image (registration error)"
    )
    print("\nshare of tiles scoring exactly 0 (clean / DEFECT -- the second is the ceiling loss):")
    for name in args.features:
        vals = [f"{r['shift']}px {r['zero_clean'] * 100:.0f}%/{r['zero_pos'] * 100:.0f}%"
                for r in rows if r["feature"] == name]
        print(f"  {name:<8}" + "   ".join(vals))
    unreachable = [r for r in rows if r["thr"] is None]
    if unreachable:
        print(f"\nNOTE: {len(unreachable)} (feature, shift) combination(s) cannot reach "
              f"{TARGET_RECALL:.0%} defect-tile recall at any threshold, because too many "
              f"defect tiles score exactly zero. Those rows show n/a rather than a "
              f"threshold that does not exist.")


if __name__ == "__main__":
    main()
