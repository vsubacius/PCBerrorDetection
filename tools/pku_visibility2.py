"""Defect visibility on PKU, per class, for both candidate Stage-1 signals.

Compares, over a threshold sweep:

  binarize-then-XOR : Otsu per channel on template and test, XOR, 3x3 erosion
                      (what the firmware does on DeepPCB)
  difference        : |test - template| on one channel > t, 3x3 erosion
                      (the brief's fallback)

Reports, per class, the share of annotated boxes with >= 1 surviving pixel, and
the noise floor outside every box, expressed as pixels per 160x160 tile -- the
number the gate threshold has to clear.

Usage:
    python -I scratchpad/pku_visibility2.py --root <clone> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

NAME_RE = re.compile(r"^(\d+)_([a-z_]+)_(\d+)$")
CLASS_DIRS = ("Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper")


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    return int(np.nan_to_num(sigma_b).argmax())


def chan(arr: np.ndarray, which: str) -> np.ndarray:
    if which == "gray":
        return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    return arr[..., "rgb".index(which)]


def erode3(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode="edge")
    out = p[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
    return out


def read_voc(path: Path):
    root = ET.parse(path).getroot()
    out = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        out.append((int(float(b.findtext("xmin"))), int(float(b.findtext("ymin"))),
                    int(float(b.findtext("xmax"))), int(float(b.findtext("ymax"))),
                    (obj.findtext("name") or "").strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--channel", default="g")
    ap.add_argument("--thresholds", default="8,12,16,20,25,30")
    args = ap.parse_args()

    root = Path(args.root)
    thresholds = [int(t) for t in args.thresholds.split(",")]
    rng = np.random.default_rng(0)

    templates = {}
    for tp in sorted((root / "PCB_USED").glob("*.JPG")):
        with Image.open(tp) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        g = chan(arr, args.channel)
        templates[tp.stem] = (arr, otsu(g))

    # [threshold][class] -> hits, and the XOR path under key "xor"
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    noise = defaultdict(list)

    for cls_dir in CLASS_DIRS:
        imgs = sorted((root / "images" / cls_dir).glob("*.jpg"))
        pick = rng.choice(len(imgs), size=min(args.per_class, len(imgs)), replace=False)
        for i in sorted(pick):
            ip = imgs[int(i)]
            board = NAME_RE.match(ip.stem).group(1)
            temp_arr, thr = templates[board]
            with Image.open(ip) as im:
                test_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if test_arr.shape != temp_arr.shape:
                continue
            tg, pg = chan(test_arr, args.channel), chan(temp_arr, args.channel)
            d = np.abs(tg.astype(np.int16) - pg.astype(np.int16))
            boxes = read_voc(root / "Annotations" / cls_dir / f"{ip.stem}.xml")

            masks = {"xor": erode3(((tg > thr) ^ (pg > thr)))}
            for t in thresholds:
                masks[t] = erode3(d > t)

            covered = np.zeros(tg.shape, dtype=bool)
            for x1, y1, x2, y2, _ in boxes:
                covered[y1:y2, x1:x2] = True
            for key, m in masks.items():
                for x1, y1, x2, y2, name in boxes:
                    h = hits[key][name]
                    h[0] += 1
                    h[1] += int(m[y1:y2, x1:x2].any())
                noise[key].append(float(m[~covered].mean()))

    report = {"channel": args.channel, "per_class": {}, "noise_per_tile": {}}
    classes = sorted(hits[thresholds[0]])
    print(f"\nchannel {args.channel}: share of annotated defects with >=1 pixel "
          f"surviving a 3x3 erosion\n")
    header = f"{'signal':<16}" + "".join(f"{c.replace('_', ' '):>16}" for c in classes) + f"{'noise px/tile':>15}"
    print(header)
    for key in ["xor"] + thresholds:
        row = []
        for c in classes:
            n, k = hits[key][c]
            row.append(k / n if n else float("nan"))
        npx = float(np.mean(noise[key])) * 160 * 160
        label = "binarize+XOR" if key == "xor" else f"|diff| > {key}"
        print(f"{label:<16}" + "".join(f"{v:>15.1%}" for v in row) + f"{npx:>15.1f}")
        report["per_class"][str(key)] = {c: (hits[key][c][1] / hits[key][c][0]) for c in classes}
        report["noise_per_tile"][str(key)] = npx

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
