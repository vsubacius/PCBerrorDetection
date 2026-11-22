"""DeepPCB reference numbers, for the PKU comparison in Step 1 of the brief.

Read-only. Nothing here touches the repo's pipeline; it only imports it.

Usage:
    python -I scratchpad/deeppcb_stats.py --repo <repo> --root <repo>/data/DeepPCB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_repo(repo: Path):
    sys.path.insert(0, str(repo / "data"))
    sys.path.insert(0, str(repo / "eval"))
    import deeppcb  # noqa: E402
    from tile_dataset import binary_morph  # noqa: E402
    from tiling import TileGrid  # noqa: E402
    return deeppcb, binary_morph, TileGrid


def stroke_widths(img: np.ndarray, binary_morph, max_radius: int = 8) -> list[float]:
    """Survival fraction of copper pixels under erosion radius 1..max_radius.

    A copper pixel surviving radius r sits in a run at least 2r+1 px wide, so
    the radius where survival crosses 50% is a median half-width. This is the
    scale proxy used to compare two datasets without knowing either's px/mm.
    """
    total = float(img.sum())
    if total == 0:
        return [0.0] * max_radius
    return [float(binary_morph(img, dilate=False, radius=r).sum()) / total
            for r in range(1, max_radius + 1)]


def run_length_stats(img: np.ndarray, max_rows: int = 640) -> dict:
    """Median copper and substrate run lengths along scanlines, in px.

    Erosion survival is dominated by large copper planes on these boards, so it
    is a poor scale proxy. Run lengths are not: the median black run is the
    board's clearance and the median white run its trace width, both physical
    distances. Runs touching a row's edge are dropped (their true length is
    unknown). This is the quantity to match between the two datasets.
    """
    rows = img[:: max(1, len(img) // max_rows)]
    white: list[int] = []
    black: list[int] = []
    for row in rows:
        d = np.diff(row.astype(np.int8))
        edges = np.flatnonzero(d) + 1
        bounds = np.concatenate([[0], edges, [len(row)]])
        lengths = np.diff(bounds)
        values = row[bounds[:-1]]
        if len(lengths) > 2:                      # drop the two border runs
            lengths, values = lengths[1:-1], values[1:-1]
            white.extend(lengths[values == 1].tolist())
            black.extend(lengths[values == 0].tolist())
    out = {}
    for name, v in (("copper", white), ("substrate", black)):
        a = np.asarray(v) if v else np.zeros(1)
        out[name] = {
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "n": int(len(a)),
        }
    return out


def median_width_px(survival: list[float]) -> float:
    """Interpolate the erosion radius at which half the copper is gone -> width."""
    prev_r, prev_s = 0, 1.0
    for r, s in enumerate(survival, start=1):
        if s < 0.5:
            # linear interpolation in r between (prev_r, prev_s) and (r, s)
            frac = (prev_s - 0.5) / (prev_s - s) if prev_s > s else 0.0
            r_half = prev_r + frac
            return 2.0 * r_half + 1.0
        prev_r, prev_s = r, s
    return 2.0 * len(survival) + 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--sample", type=int, default=40, help="images for stroke width")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(args.repo)
    deeppcb, binary_morph, TileGrid = load_repo(repo)

    samples = deeppcb.index(args.root, None)
    print(f"DeepPCB: {len(samples)} pairs, "
          f"{sum(s.split == 'trainval' for s in samples)} trainval / "
          f"{sum(s.split == 'test' for s in samples)} test")
    print(f"groups (physical boards): {len(set(s.group for s in samples))}")

    # --- boxes: count and size per class -------------------------------------
    per_class: dict[str, list[tuple[int, int]]] = {}
    n_boxes = 0
    for s in samples:
        for x1, y1, x2, y2, cls in s.boxes():
            name = deeppcb.CLASS_NAMES[int(cls)]
            per_class.setdefault(name, []).append((int(x2 - x1), int(y2 - y1)))
            n_boxes += 1
    print(f"\nboxes: {n_boxes}")
    print(f"{'class':<12}{'n':>7}{'short side med':>16}{'long side med':>15}"
          f"{'short p90':>11}{'max':>7}")
    rows = {}
    for name, wh in sorted(per_class.items()):
        a = np.asarray(wh)
        short = np.minimum(a[:, 0], a[:, 1])
        long_ = np.maximum(a[:, 0], a[:, 1])
        rows[name] = {
            "n": len(a),
            "short_median": float(np.median(short)),
            "long_median": float(np.median(long_)),
            "short_p90": float(np.percentile(short, 90)),
            "max_side": int(long_.max()),
        }
        print(f"{name:<12}{len(a):>7}{np.median(short):>16.1f}{np.median(long_):>15.1f}"
              f"{np.percentile(short, 90):>11.1f}{long_.max():>7}")
    allshort = np.concatenate([np.minimum(np.asarray(v)[:, 0], np.asarray(v)[:, 1])
                               for v in per_class.values()])
    print(f"{'ALL':<12}{len(allshort):>7}{np.median(allshort):>16.1f}")

    # --- scale proxy: copper stroke width ------------------------------------
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(args.sample, len(samples)), replace=False)
    surv = []
    for i in pick:
        temp = deeppcb.load_binary(samples[int(i)].temp_path)
        surv.append(stroke_widths(temp, binary_morph))
    surv_mean = np.asarray(surv).mean(axis=0).tolist()
    width = median_width_px(surv_mean)
    print(f"\ncopper stroke width (templates, n={len(pick)}):")
    print("  erosion radius:      " + "".join(f"{r:>8}" for r in range(1, 9)))
    print("  copper surviving:    " + "".join(f"{s:>7.1%}" for s in surv_mean))
    print(f"  median trace width:  {width:.1f} px (fallback value if survival never halves)")

    rl = [run_length_stats(deeppcb.load_binary(samples[int(i)].temp_path)) for i in pick[:15]]
    rl_med = {k: {m: float(np.mean([r[k][m] for r in rl])) for m in ("p10", "median", "p90")}
              for k in ("copper", "substrate")}
    print(f"\nscanline run lengths (templates, n={len(rl)}), px:")
    for k, v in rl_med.items():
        print(f"  {k:<10} p10 {v['p10']:>6.1f}   median {v['median']:>6.1f}   p90 {v['p90']:>7.1f}")
    print(f"  copper fill fraction: "
          f"{np.mean([deeppcb.load_binary(samples[int(i)].temp_path).mean() for i in pick[:10]]):.3f}")

    out = {
        "n_pairs": len(samples),
        "n_boxes": n_boxes,
        "per_class": rows,
        "short_side_median_all": float(np.median(allshort)),
        "erosion_survival": surv_mean,
        "median_trace_width_px": width,
        "run_lengths_px": rl_med,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
