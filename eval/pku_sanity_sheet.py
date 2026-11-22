"""Look at what the PKU converter produced, before trusting any number from it.

One row per crop: template, test with its boxes drawn, and the eroded XOR the
Stage-1 gate actually counts. A conversion can be wrong in ways no summary
statistic shows -- boxes off by a scale factor, a template paired with the
wrong board, a binarization that erases the traces -- and all three are
obvious here in a second.

Usage::

    python eval/pku_sanity_sheet.py --n 10 --out eval/figures_pku/sanity.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))

import pku  # noqa: E402
from tile_dataset import binary_morph  # noqa: E402
from tiling import TileGrid  # noqa: E402

RED = np.array([220, 40, 40], dtype=np.uint8)
GREEN = np.array([40, 200, 90], dtype=np.uint8)
GREY = np.array([90, 90, 90], dtype=np.uint8)


def as_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)


def draw_box(img: np.ndarray, box, colour=RED, width: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    for w in range(width):
        img[np.clip(y1 + w, 0, img.shape[0] - 1), x1:x2] = colour
        img[np.clip(y2 - 1 - w, 0, img.shape[0] - 1), x1:x2] = colour
        img[y1:y2, np.clip(x1 + w, 0, img.shape[1] - 1)] = colour
        img[y1:y2, np.clip(x2 - 1 - w, 0, img.shape[1] - 1)] = colour


def draw_grid(img: np.ndarray, grid: TileGrid) -> None:
    """Faint tile boundaries, so defect size can be read against the tiles."""
    for x1, y1, x2, y2 in grid.tiles():
        img[y1, x1:x2] = GREY
        img[y1:y2, x1] = GREY


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/PKU-crops")
    ap.add_argument("--out", default="eval/figures_pku/pku_sanity_sheet.png")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--clean", type=int, default=2, help="also show N clean crops")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.5, help="output downscale")
    args = ap.parse_args()

    samples = pku.index(args.root)
    with_boxes = [s for s in samples if len(s.boxes())]
    clean = [s for s in samples if not len(s.boxes())]
    rng = np.random.default_rng(args.seed)

    chosen = [with_boxes[int(i)] for i in
              rng.choice(len(with_boxes), size=min(args.n, len(with_boxes)),
                         replace=False)]
    if args.clean and clean:
        chosen += [clean[int(i)] for i in
                   rng.choice(len(clean), size=min(args.clean, len(clean)),
                              replace=False)]

    grid = TileGrid()
    rows = []
    print(f"{'crop':<44}{'boxes':>6}{'copper':>9}{'xor px':>9}{'eroded':>8}")
    for s in chosen:
        test, temp = s.load()
        boxes = s.boxes()
        xor = (test ^ temp).astype(np.uint8)
        eroded = binary_morph(xor, dilate=False, radius=args.radius)

        temp_rgb, test_rgb, xor_rgb = as_rgb(temp), as_rgb(test), as_rgb(eroded)
        draw_grid(test_rgb, grid)
        for box in boxes:
            draw_box(test_rgb, box, RED)
            draw_box(xor_rgb, box, GREEN)
        rows.append(np.concatenate([temp_rgb, test_rgb, xor_rgb], axis=1))
        print(f"{s.stem:<44}{len(boxes):>6}{test.mean():>8.1%}"
              f"{int(xor.sum()):>9}{int(eroded.sum()):>8}")

    sheet = np.concatenate(rows, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(sheet)
    if args.scale != 1.0:
        img = img.resize((int(img.width * args.scale), int(img.height * args.scale)),
                         Image.BOX)
    img.save(out)
    print(f"\nwrote {out}  ({img.width}x{img.height})")
    print("columns: template | test (red boxes, grey tile grid) | eroded XOR "
          "(green boxes)")


if __name__ == "__main__":
    main()
