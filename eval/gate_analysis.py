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
* ``thr@98``        -- feature threshold that keeps 98% of defect tiles
* ``skip@98``       -- share of *clean* tiles skipped at that threshold; this is
                       the Stage-2 compute the cascade actually saves
* ``recall@skip50`` -- defect-tile recall if the gate is tuned to skip half the
                       clean tiles instead

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


def analyse(data: TileData, features=("raw", "erode1", "erode2"),
            shifts=(0, 1, 2, 3), seed: int = 0) -> list[dict]:
    rows = []
    for feature in features:
        for shift in shifts:
            scores, pos, clean = tile_scores(data, feature, shift, seed)
            s_pos, s_neg = scores[pos], scores[clean]
            # The gate fires when score > thr, so the threshold that keeps
            # TARGET_RECALL of defect tiles is the low quantile of s_pos.
            thr = float(np.quantile(s_pos, 1.0 - TARGET_RECALL))
            thr2 = float(np.quantile(s_neg, TARGET_SKIP))
            rows.append({
                "feature": feature, "shift": shift,
                "n_pos": len(s_pos), "n_clean": len(s_neg),
                "median_pos": float(np.median(s_pos)),
                "median_clean": float(np.median(s_neg)),
                "zero_clean": float((s_neg == 0).mean()),
                "auc": auc(s_pos, s_neg),
                "thr": thr, "skip": float((s_neg <= thr).mean()),
                "recall_skip50": float((s_pos > thr2).mean()),
            })
    return rows


def format_table(rows: list[dict]) -> str:
    head = (f"{'feature':>8}{'shift':>7}{'med clean':>11}{'med defect':>12}"
            f"{'AUC':>8}{'thr@98':>8}{'skip@98':>9}{'recall@skip50':>15}")
    out = [head, "-" * len(head)]
    last = None
    for r in rows:
        if last is not None and r["feature"] != last:
            out.append("")
        last = r["feature"]
        out.append(
            f"{r['feature']:>8}{str(r['shift']) + 'px':>7}{r['median_clean']:>11.0f}"
            f"{r['median_pos']:>12.0f}{r['auc']:>8.4f}{r['thr']:>8.0f}"
            f"{r['skip'] * 100:>8.1f}%{r['recall_skip50'] * 100:>14.1f}%"
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
        f"\nthr@{TARGET_RECALL:.0%}        = threshold that keeps {TARGET_RECALL:.0%} of defect tiles"
        f"\nskip@{TARGET_RECALL:.0%}       = share of clean tiles skipped there (= Stage-2 compute saved)"
        f"\nrecall@skip{TARGET_SKIP:.0%}   = defect-tile recall when tuned to skip {TARGET_SKIP:.0%} of clean tiles"
        "\nshift           = template jittered up to +-N px per image (registration error)"
    )
    print("\nshare of clean tiles scoring exactly 0:")
    for name in args.features:
        vals = [f"{r['shift']}px {r['zero_clean'] * 100:.1f}%"
                for r in rows if r["feature"] == name]
        print(f"  {name:<8}" + "   ".join(vals))


if __name__ == "__main__":
    main()
