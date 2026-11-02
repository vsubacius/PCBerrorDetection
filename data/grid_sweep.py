"""Pick a tile/stride by measuring real defect containment, not by guessing.

A defect can only ever be found if some tile holds at least ``pos_thresh`` of
its area -- otherwise no tile is labelled with its class and the defect is
invisible to the cascade no matter how good Stage 2 is. That makes
*reachability* a hard ceiling on the headline defect-level recall, and it is
decided entirely by tile size, stride and ``pos_thresh``.

Run from the repo root::

    python data/grid_sweep.py --root data/DeepPCB

Prints one row per candidate grid: the recall ceiling, the share of tiles that
are clean (what the Stage-1 gate can skip) and the Stage-2 pixel cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deeppcb  # noqa: E402
from deeppcb import CLASS_IDS, CLASS_NAMES  # noqa: E402
from tiling import POS_THRESH, TileGrid, box_overlap_fracs, is_clean, label_tiles  # noqa: E402

#: (tile, stride) candidates. 640 = 5*128, so several of these land on a 5x5
#: grid with different amounts of overlap.
DEFAULT_CANDIDATES: tuple[tuple[int, int], ...] = (
    (128, 128),   # 5x5, no overlap -- cheapest possible
    (160, 160),   # 4x4, no overlap
    (160, 120),   # 5x5, 40 px overlap  (the original proposal)
    (160, 96),    # 6x6, 64 px overlap
    (160, 80),    # 7x7, 80 px overlap
    (192, 112),   # 5x5, 80 px overlap
    (192, 96),    # 6x6, 96 px overlap
    (224, 104),   # 5x5, 120 px overlap
    (256, 96),    # 5x5, 160 px overlap
)


def collect_boxes(samples) -> np.ndarray:
    """Stack every annotation box with a trailing sample index column."""
    out = []
    for i, s in enumerate(samples):
        b = s.boxes()
        if len(b):
            out.append(np.column_stack([b, np.full(len(b), i, dtype=np.int32)]))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 6), dtype=np.int32)


def box_stats(boxes: np.ndarray) -> str:
    lines = ["defect box sizes (max of width/height, pixels):",
             f"  {'class':<12}{'n':>6}{'mean':>7}{'p50':>6}{'p90':>6}{'p99':>6}{'max':>6}"]
    maxdim = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    for cid in CLASS_IDS:
        m = maxdim[boxes[:, 4] == cid]
        lines.append(
            f"  {CLASS_NAMES[cid]:<12}{len(m):>6}{m.mean():>7.1f}"
            f"{np.percentile(m, 50):>6.0f}{np.percentile(m, 90):>6.0f}"
            f"{np.percentile(m, 99):>6.0f}{m.max():>6}"
        )
    lines.append(
        f"  {'ALL':<12}{len(maxdim):>6}{maxdim.mean():>7.1f}"
        f"{np.percentile(maxdim, 50):>6.0f}{np.percentile(maxdim, 90):>6.0f}"
        f"{np.percentile(maxdim, 99):>6.0f}{maxdim.max():>6}"
    )
    return "\n".join(lines)


def evaluate_grid(
    boxes: np.ndarray,
    samples,
    grid: TileGrid,
    pos_thresh: float = POS_THRESH,
    tile_label_subset: int = 200,
) -> dict:
    """Containment + tile-population stats for one grid.

    Reachability uses *every* box (cheap: pure geometry). Tile-population
    stats only need a subset of images, since they are averages.
    """
    tiles = grid.tiles()
    fracs = box_overlap_fracs(boxes, tiles)
    best = fracs.max(axis=1)

    per_class = {}
    for cid in CLASS_IDS:
        m = boxes[:, 4] == cid
        per_class[CLASS_NAMES[cid]] = float((best[m] >= pos_thresh).mean())

    sub = samples[:tile_label_subset]
    n_clean = n_tiles_total = n_ign = 0
    n_pos = 0
    for s in sub:
        y, ign = label_tiles(s.boxes(), grid, pos_thresh=pos_thresh)
        n_tiles_total += len(y)
        n_clean += int(is_clean(y, ign).sum())
        n_ign += int(ign.any(axis=1).sum())
        n_pos += int((y > 0).any(axis=1).sum())

    return {
        "grid": grid,
        "reachable": float((best >= pos_thresh).mean()),
        "fully_contained": float((best >= 1.0).mean()),
        "per_class": per_class,
        "clean_frac": n_clean / n_tiles_total,
        "pos_frac": n_pos / n_tiles_total,
        "ignore_frac": n_ign / n_tiles_total,
    }


def format_table(rows: list[dict], pos_thresh: float) -> str:
    head = (
        f"{'tile/stride':>12}{'grid':>8}{'ovl':>5}{'tiles':>7}"
        f"{'px cost':>9}{'reach@%.2f' % pos_thresh:>12}{'full':>8}"
        f"{'clean':>8}{'pos':>7}{'ign':>7}"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        g: TileGrid = r["grid"]
        rr, cc = g.grid_shape
        out.append(
            f"{f'{g.tile}/{g.stride}':>12}{f'{rr}x{cc}':>8}{g.overlap:>5}{g.n_tiles:>7}"
            f"{g.pixel_cost / g.image_size**2:>8.2f}x{r["reachable"] * 100:>10.2f}%"
            f"{r['fully_contained'] * 100:>7.1f}%{r['clean_frac'] * 100:>7.1f}%"
            f"{r['pos_frac'] * 100:>6.1f}%{r['ignore_frac'] * 100:>6.1f}%"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/DeepPCB", help="DeepPCB clone directory")
    ap.add_argument("--split", default="trainval", choices=["trainval", "test", "all"])
    ap.add_argument("--pos-thresh", type=float, default=POS_THRESH)
    ap.add_argument("--download", action="store_true", help="clone the dataset if missing")
    args = ap.parse_args()

    if args.download:
        deeppcb.download(args.root)
    samples = deeppcb.index(args.root, None if args.split == "all" else args.split)
    boxes = collect_boxes(samples)
    print(f"{len(samples)} images, {len(boxes)} defect boxes "
          f"({len(boxes) / len(samples):.2f} per image)\n")
    print(box_stats(boxes), "\n")

    rows = [
        evaluate_grid(boxes, samples, TileGrid(t, s), args.pos_thresh)
        for t, s in DEFAULT_CANDIDATES
    ]
    print(format_table(rows, args.pos_thresh))
    print(
        "\nreach = share of defects with >= pos_thresh of their area in some tile"
        "\n        (a hard ceiling on defect-level recall)"
        "\nfull  = share of defects fully inside some tile"
        "\nclean/pos/ign = share of tiles that are clean / hold a positive class /"
        "\n        hold only a fragment (first 200 images)"
        "\npx cost = Stage-2 pixels per board with the gate off, relative to 640x640"
    )

    print("\nper-class reachability (%):")
    names = [CLASS_NAMES[c] for c in CLASS_IDS]
    print(f"{'tile/stride':>12}" + "".join(f"{n:>11}" for n in names))
    for r in rows:
        g: TileGrid = r["grid"]
        print(
            f"{f'{g.tile}/{g.stride}':>12}"
            + "".join(f"{r['per_class'][n] * 100:>10.1f}%" for n in names)
        )


if __name__ == "__main__":
    main()
