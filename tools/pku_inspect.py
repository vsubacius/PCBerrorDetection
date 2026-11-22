"""Step 1 of PKU_SWITCH_BRIEF.md: inspect PKU-Market-PCB and report.

Read-only with respect to the dataset and the repo pipeline. Writes only the
JSON report and the sanity crops it is told to write.

Usage:
    python -I scratchpad/pku_inspect.py --root <clone> --out-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

#: PKU class directory -> the brief's lowercase class name.
PKU_CLASSES = {
    "Missing_hole": "missing_hole",
    "Mouse_bite": "mouse_bite",
    "Open_circuit": "open_circuit",
    "Short": "short",
    "Spur": "spur",
    "Spurious_copper": "spurious_copper",
}

NAME_RE = re.compile(r"^(\d+)_([a-z_]+)_(\d+)$")


@dataclass(frozen=True)
class PkuSample:
    board: str          # "01" .. "12", the PCB_USED template id
    cls_dir: str        # "Missing_hole"
    stem: str           # "01_missing_hole_01"
    image: Path
    ann: Path

    @property
    def template(self) -> str:
        return self.board


def index(root: Path) -> list[PkuSample]:
    out: list[PkuSample] = []
    for cls_dir in sorted(PKU_CLASSES):
        img_dir = root / "images" / cls_dir
        ann_dir = root / "Annotations" / cls_dir
        for img in sorted(img_dir.glob("*.jpg")):
            m = NAME_RE.match(img.stem)
            if not m:
                raise ValueError(f"unexpected image name: {img}")
            ann = ann_dir / f"{img.stem}.xml"
            if not ann.exists():
                raise FileNotFoundError(f"no annotation for {img}")
            out.append(PkuSample(m.group(1), cls_dir, img.stem, img, ann))
    return out


def read_voc(path: Path) -> tuple[tuple[int, int], list[tuple[int, int, int, int, str]]]:
    """VOC XML -> ((width, height), [(x1, y1, x2, y2, name), ...])."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1, y1, x2, y2, name))
    return wh, boxes


# --------------------------------------------------------------------------- #
# binarization (preliminary -- Step 2 replaces this with the firmware's copy)
# --------------------------------------------------------------------------- #

def otsu(gray: np.ndarray) -> tuple[int, float]:
    """Global Otsu threshold and the between-class variance ratio it achieves.

    The ratio (between-class / total variance) is a bimodality score: how
    cleanly the channel splits copper from laminate. Used to pick a channel.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b = np.nan_to_num(sigma_b)
    k = int(sigma_b.argmax())
    var_total = float((p * (np.arange(256) - mu_t) ** 2).sum())
    return k, float(sigma_b[k] / var_total) if var_total else 0.0


def binarize(img: np.ndarray, channel: str = "gray") -> np.ndarray:
    """Pick a channel, Otsu it, return {0,1} with copper = 1.

    Copper is assumed to be the brighter class under green solder mask on these
    photos; the polarity is asserted by checking the copper fraction is < 0.5
    is NOT enforced here -- the caller reports the fraction instead.
    """
    if img.ndim == 3:
        if channel == "gray":
            g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
            g = g.astype(np.uint8)
        else:
            g = img[..., "rgb".index(channel)]
    else:
        g = img
    t, _ = otsu(g)
    return (g > t).astype(np.uint8)


def run_length_stats(img: np.ndarray, max_rows: int = 640) -> dict:
    """Median copper / substrate run lengths along scanlines, in px."""
    step = max(1, len(img) // max_rows)
    white: list[int] = []
    black: list[int] = []
    for row in img[::step]:
        d = np.diff(row.astype(np.int8))
        bounds = np.concatenate([[0], np.flatnonzero(d) + 1, [len(row)]])
        lengths = np.diff(bounds)
        values = row[bounds[:-1]]
        if len(lengths) > 2:
            lengths, values = lengths[1:-1], values[1:-1]
            white.extend(lengths[values == 1].tolist())
            black.extend(lengths[values == 0].tolist())
    out = {}
    for name, v in (("copper", white), ("substrate", black)):
        a = np.asarray(v) if v else np.zeros(1)
        out[name] = {"p10": float(np.percentile(a, 10)),
                     "median": float(np.median(a)),
                     "p90": float(np.percentile(a, 90)),
                     "n": int(len(a))}
    return out


def best_shift(test: np.ndarray, temp: np.ndarray, radius: int = 8) -> tuple[int, int, int, int]:
    """Integer (dx, dy) minimizing XOR count, plus the XOR at 0 and at best.

    Shifts the test image; compares on the overlap only, normalized per pixel so
    different overlaps are comparable. Same method as the firmware's alignment.
    """
    h, w = temp.shape
    base = None
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys = slice(max(0, dy), h + min(0, dy))
            xs = slice(max(0, dx), w + min(0, dx))
            yt = slice(max(0, -dy), h + min(0, -dy))
            xt = slice(max(0, -dx), w + min(0, -dx))
            a = test[yt, xt]
            b = temp[ys, xs]
            cost = float((a ^ b).mean())
            if dx == 0 and dy == 0:
                base = cost
            if best is None or cost < best[2]:
                best = (dx, dy, cost)
    return best[0], best[1], base, best[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--align-sample", type=int, default=20)
    ap.add_argument("--align-radius", type=int, default=8)
    ap.add_argument("--align-crop", type=int, default=600)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    samples = index(root)
    print(f"images: {len(samples)}")

    # --- 1. counts, boards, sizes, boxes per class ---------------------------
    per_class = Counter(s.cls_dir for s in samples)
    per_board = Counter(s.board for s in samples)
    templates = sorted(p.stem for p in (root / "PCB_USED").glob("*.JPG"))
    print(f"templates in PCB_USED: {len(templates)} -> {', '.join(templates)}")
    missing_templates = sorted(set(per_board) - set(templates))
    print(f"boards referenced by images: {len(per_board)}; "
          f"without a template: {missing_templates or 'none'}")

    sizes = Counter()
    boxes_by_class: dict[str, list[tuple[int, int]]] = defaultdict(list)
    xml_names = Counter()
    boxes_per_image = []
    board_sizes: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        wh, boxes = read_voc(s.ann)
        sizes[wh] += 1
        board_sizes[s.board][wh] += 1
        boxes_per_image.append(len(boxes))
        for x1, y1, x2, y2, name in boxes:
            xml_names[name] += 1
            boxes_by_class[name].append((x2 - x1, y2 - y1))

    print("\nimages per class dir:")
    for k, v in sorted(per_class.items()):
        print(f"  {k:<18}{v:>5}")
    print("\nimages per board (template):")
    print("  " + "  ".join(f"{b}:{n}" for b, n in sorted(per_board.items())))
    print("\nimage sizes (from the annotations):")
    for wh, n in sizes.most_common():
        print(f"  {wh[0]}x{wh[1]}{'':<4}{n:>5} images")
    print("\nsize per board:")
    for b in sorted(board_sizes):
        print(f"  {b}: " + ", ".join(f"{w}x{h} x{n}" for (w, h), n in board_sizes[b].most_common()))

    n_boxes = sum(xml_names.values())
    print(f"\nboxes: {n_boxes}  (mean {np.mean(boxes_per_image):.2f} per image, "
          f"min {min(boxes_per_image)}, max {max(boxes_per_image)})")
    print(f"{'xml class name':<20}{'n':>7}{'short med':>11}{'long med':>10}{'short p90':>11}{'max':>7}")
    class_rows = {}
    for name, wh in sorted(boxes_by_class.items()):
        a = np.asarray(wh)
        short = np.minimum(a[:, 0], a[:, 1])
        long_ = np.maximum(a[:, 0], a[:, 1])
        class_rows[name] = {"n": len(a),
                            "short_median": float(np.median(short)),
                            "long_median": float(np.median(long_)),
                            "short_p90": float(np.percentile(short, 90)),
                            "max_side": int(long_.max())}
        print(f"{name:<20}{len(a):>7}{np.median(short):>11.1f}{np.median(long_):>10.1f}"
              f"{np.percentile(short, 90):>11.1f}{long_.max():>7}")

    report.update(
        n_images=len(samples),
        images_per_class={k: v for k, v in sorted(per_class.items())},
        images_per_board={k: v for k, v in sorted(per_board.items())},
        templates=templates,
        boards_without_template=missing_templates,
        image_sizes={f"{w}x{h}": n for (w, h), n in sizes.most_common()},
        n_boxes=n_boxes,
        boxes_per_image={"mean": float(np.mean(boxes_per_image)),
                         "min": int(min(boxes_per_image)),
                         "max": int(max(boxes_per_image))},
        boxes_by_class=class_rows,
    )

    # --- 2. channel choice: which channel separates copper from laminate -----
    print("\nchannel bimodality on the 10 templates (Otsu between/total variance):")
    print(f"{'board':<8}" + "".join(f"{c:>10}" for c in ("gray", "r", "g", "b")))
    chan_scores: dict[str, list[float]] = defaultdict(list)
    chan_thresh: dict[str, list[int]] = defaultdict(list)
    copper_frac: dict[str, float] = {}
    for b in templates:
        with Image.open(root / "PCB_USED" / f"{b}.JPG") as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        row = []
        for c in ("gray", "r", "g", "b"):
            if c == "gray":
                g = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
            else:
                g = arr[..., "rgb".index(c)]
            t, score = otsu(g)
            chan_scores[c].append(score)
            chan_thresh[c].append(t)
            row.append(score)
        print(f"{b:<8}" + "".join(f"{v:>10.3f}" for v in row))
        copper_frac[b] = float(binarize(arr, "gray").mean())
    print(f"{'mean':<8}" + "".join(f"{np.mean(chan_scores[c]):>10.3f}" for c in ("gray", "r", "g", "b")))
    print("  Otsu thresholds (gray): " + ", ".join(str(t) for t in chan_thresh["gray"]))
    print("  bright-class fraction per template (gray): "
          + ", ".join(f"{b}:{copper_frac[b]:.2f}" for b in templates))
    report["channel_bimodality"] = {c: float(np.mean(chan_scores[c])) for c in ("gray", "r", "g", "b")}
    report["bright_fraction_gray"] = copper_frac

    # --- 3. scale: run lengths on the templates -----------------------------
    print("\nscanline run lengths on binarized templates (px):")
    rl_all = {}
    for b in templates:
        with Image.open(root / "PCB_USED" / f"{b}.JPG") as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        bw = binarize(arr, "gray")
        rl = run_length_stats(bw)
        rl_all[b] = rl
        print(f"  {b}: copper med {rl['copper']['median']:>5.1f}  p90 {rl['copper']['p90']:>6.1f}"
              f"   substrate med {rl['substrate']['median']:>5.1f}  p90 {rl['substrate']['p90']:>6.1f}")
    for k in ("copper", "substrate"):
        med = float(np.median([rl_all[b][k]["median"] for b in templates]))
        print(f"  median over boards, {k}: {med:.1f} px")
        report.setdefault("run_lengths_px", {})[k] = med

    # --- 4. alignment residual ----------------------------------------------
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(args.align_sample, len(samples)), replace=False)
    print(f"\nalignment residual, {len(pick)} pairs, +-{args.align_radius} px search "
          f"on a {args.align_crop}px central crop:")
    shifts = []
    for i in sorted(pick):
        s = samples[int(i)]
        with Image.open(s.image) as im:
            test = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(root / "PCB_USED" / f"{s.board}.JPG") as im:
            temp = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if test.shape != temp.shape:
            print(f"  {s.stem}: SIZE MISMATCH test {test.shape[:2]} vs template {temp.shape[:2]}")
            continue
        tb = binarize(test, "gray")
        pb = binarize(temp, "gray")
        h, w = tb.shape
        c = args.align_crop
        y0, x0 = (h - c) // 2, (w - c) // 2
        dx, dy, base, best = best_shift(tb[y0:y0 + c, x0:x0 + c], pb[y0:y0 + c, x0:x0 + c],
                                        args.align_radius)
        shifts.append((s.stem, s.board, dx, dy, base, best))
        print(f"  {s.stem:<26} board {s.board}  dx {dx:>3} dy {dy:>3}   "
              f"XOR {base:.4f} -> {best:.4f}")
    if shifts:
        mags = [max(abs(s[2]), abs(s[3])) for s in shifts]
        print(f"  |shift| median {np.median(mags):.1f} px, max {max(mags)} px, "
              f"at search radius {args.align_radius}")
        print(f"  XOR density at zero shift: median {np.median([s[4] for s in shifts]):.4f}, "
              f"after alignment {np.median([s[5] for s in shifts]):.4f}")
        report["alignment"] = {
            "n_pairs": len(shifts),
            "search_radius": args.align_radius,
            "shift_median": float(np.median(mags)),
            "shift_max": int(max(mags)),
            "xor_zero_shift_median": float(np.median([s[4] for s in shifts])),
            "xor_aligned_median": float(np.median([s[5] for s in shifts])),
            "per_pair": [{"stem": s[0], "board": s[1], "dx": s[2], "dy": s[3],
                          "xor0": s[4], "xor_best": s[5]} for s in shifts],
        }

    out = out_dir / "pku_inspect.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
