"""Build the labelled tile dataset from a DeepPCB clone.

Writes a single ``.npz`` holding, for every image pair:

* ``images_test`` / ``images_temp`` -- the binarized 640x640 images,
  **bit-packed** along x to ``(N, 640, 80) uint8``
* ``y`` ``(N, n_tiles, 6) uint8`` and ``ignore`` ``(N, n_tiles, 6) uint8``
  -- the per-tile multi-label targets and ignore masks
* ``boxes`` ``(M, 5) int32`` + ``box_image`` ``(M,) int32`` -- the raw
  annotations, kept for the defect-level recall metric
* ``fold`` -- ``train`` / ``val`` / ``test`` per image
* ``meta`` -- a JSON blob with the grid config and every threshold used

Full images are stored rather than pre-cut tiles for two reasons: overlapping
tiles would duplicate 1.56x of the pixels, and keeping the images whole means
the tile size can be changed (it is an ablation axis) without rebuilding.
Bit-packing exploits the fact that the images are binary -- the whole dataset
is ~154 MB, so it fits in Colab RAM and caches on Drive.

The train/val split is by **image**, never by tile: neighbouring tiles overlap
by 40 px, so a tile-level split would leak. ``--split-by group`` is stricter
still -- DeepPCB's 13 groups are different physical boards, so a group-level
split also prevents board leakage (use it to check the val score is not
optimistic).

Usage::

    python data/build_tiles.py --root data/DeepPCB --out data/tiles --download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deeppcb  # noqa: E402
from deeppcb import CLASS_IDS, CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES  # noqa: E402
from tiling import IGNORE_THRESH, POS_THRESH, TileGrid, is_clean, label_tiles  # noqa: E402


def assign_folds(samples, val_frac: float, seed: int, split_by: str) -> np.ndarray:
    """``(N,) <U5`` array of ``train`` / ``val`` / ``test`` per sample.

    ``test`` is DeepPCB's official 500-image test split, untouched. ``val`` is
    carved out of the official 1,000 trainval images only.
    """
    folds = np.array([s.split for s in samples], dtype=object)
    folds = np.where(folds == "test", "test", "train").astype("<U5")

    trainval_idx = np.flatnonzero(folds == "train")
    rng = np.random.default_rng(seed)

    if split_by == "image":
        keys = np.array([samples[i].name for i in trainval_idx])
    elif split_by == "group":
        keys = np.array([samples[i].group for i in trainval_idx])
    else:
        raise ValueError(f"split_by must be 'image' or 'group', got {split_by!r}")

    unique = np.unique(keys)
    rng.shuffle(unique)
    n_val = max(1, int(round(val_frac * len(unique))))
    val_keys = set(unique[:n_val].tolist())
    folds[trainval_idx[np.isin(keys, list(val_keys))]] = "val"
    return folds


def build(
    root: str,
    out_dir: str,
    grid: TileGrid,
    pos_thresh: float = POS_THRESH,
    ignore_thresh: float = IGNORE_THRESH,
    val_frac: float = 0.2,
    seed: int = 0,
    split_by: str = "image",
    limit: int | None = None,
    binarize_threshold: int = 127,
) -> Path:
    samples = deeppcb.index(root, None)
    if limit:
        samples = samples[:limit]
    folds = assign_folds(samples, val_frac, seed, split_by)

    n = len(samples)
    packed_w = (IMAGE_SIZE + 7) // 8
    images_test = np.zeros((n, IMAGE_SIZE, packed_w), dtype=np.uint8)
    images_temp = np.zeros((n, IMAGE_SIZE, packed_w), dtype=np.uint8)
    y_all = np.zeros((n, grid.n_tiles, NUM_CLASSES), dtype=np.uint8)
    ign_all = np.zeros((n, grid.n_tiles, NUM_CLASSES), dtype=np.uint8)
    all_boxes: list[np.ndarray] = []
    box_image: list[np.ndarray] = []

    t0 = time.time()
    for i, s in enumerate(samples):
        test = deeppcb.load_binary(s.test_path, binarize_threshold)
        temp = deeppcb.load_binary(s.temp_path, binarize_threshold)
        images_test[i] = np.packbits(test, axis=-1)
        images_temp[i] = np.packbits(temp, axis=-1)

        boxes = s.boxes()
        y, ign = label_tiles(boxes, grid, pos_thresh, ignore_thresh)
        y_all[i] = y.astype(np.uint8)
        ign_all[i] = ign.astype(np.uint8)
        if len(boxes):
            all_boxes.append(boxes)
            box_image.append(np.full(len(boxes), i, dtype=np.int32))

        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} images ({time.time() - t0:.0f}s)", flush=True)

    boxes_arr = (
        np.concatenate(all_boxes).astype(np.int32)
        if all_boxes else np.zeros((0, 5), np.int32)
    )
    box_image_arr = (
        np.concatenate(box_image) if box_image else np.zeros((0,), np.int32)
    )

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "DeepPCB",
        "dataset_url": deeppcb.DEEPPCB_URL,
        "image_size": IMAGE_SIZE,
        "binarize_threshold": binarize_threshold,
        "tile": grid.tile,
        "stride": grid.stride,
        "overlap": grid.overlap,
        "grid_shape": list(grid.grid_shape),
        "n_tiles": grid.n_tiles,
        "pos_thresh": pos_thresh,
        "ignore_thresh": ignore_thresh,
        "class_ids": list(CLASS_IDS),
        "class_names": [CLASS_NAMES[c] for c in CLASS_IDS],
        "val_frac": val_frac,
        "seed": seed,
        "split_by": split_by,
        "packed_x": True,
    }

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out = out_dir_p / f"deeppcb_tiles_t{grid.tile}_s{grid.stride}.npz"
    np.savez_compressed(
        out,
        images_test=images_test,
        images_temp=images_temp,
        y=y_all,
        ignore=ign_all,
        boxes=boxes_arr,
        box_image=box_image_arr,
        fold=folds,
        name=np.array([s.name for s in samples]),
        group=np.array([s.group for s in samples]),
        official_split=np.array([s.split for s in samples]),
        meta=np.array(json.dumps(meta)),
    )

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    print(summarize(y_all, ign_all, folds, grid))
    return out


def summarize(y: np.ndarray, ign: np.ndarray, folds: np.ndarray, grid: TileGrid) -> str:
    """Human-readable label statistics, printed after a build."""
    lines = [f"\n{grid.describe()}"]
    lines.append(f"{'fold':>6}{'images':>8}{'tiles':>8}{'clean':>8}{'pos':>7}{'ign':>7}")
    for fold in ("train", "val", "test"):
        m = folds == fold
        if not m.any():
            continue
        yy = y[m].reshape(-1, y.shape[-1]).astype(np.float32)
        ii = ign[m].reshape(-1, ign.shape[-1]).astype(bool)
        n_t = len(yy)
        lines.append(
            f"{fold:>6}{int(m.sum()):>8}{n_t:>8}"
            f"{100 * is_clean(yy, ii).mean():>7.1f}%"
            f"{100 * (yy > 0).any(1).mean():>6.1f}%"
            f"{100 * ii.any(1).mean():>6.1f}%"
        )
    lines.append("\npositive tiles per class (train / val / test):")
    for c, name in enumerate([CLASS_NAMES[i] for i in CLASS_IDS]):
        counts = [int(y[folds == f][..., c].sum()) for f in ("train", "val", "test")]
        lines.append(f"  {name:<12}" + " / ".join(f"{v:>6}" for v in counts))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/DeepPCB")
    ap.add_argument("--out", default="data/tiles")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--tile", type=int, default=160)
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--pos-thresh", type=float, default=POS_THRESH)
    ap.add_argument("--ignore-thresh", type=float, default=IGNORE_THRESH)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-by", default="image", choices=["image", "group"])
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images")
    args = ap.parse_args()

    if args.download:
        deeppcb.download(args.root)
    build(
        root=args.root,
        out_dir=args.out,
        grid=TileGrid(args.tile, args.stride),
        pos_thresh=args.pos_thresh,
        ignore_thresh=args.ignore_thresh,
        val_frac=args.val_frac,
        seed=args.seed,
        split_by=args.split_by,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
