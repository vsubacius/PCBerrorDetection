"""Why do five of six PKU classes vanish? Check coordinates, then contrast.

1. Does the XML <size> match the JPEG on disk? (A mismatch would put every box
   in the wrong place and look exactly like "invisible defects".)
2. Inside each annotated box, how large is |test - template| really, per
   channel, before any threshold? This says whether the defect is low-contrast
   or absent.
3. Writes a crop sheet: template | test | |diff| amplified, per class.

Usage:
    python -I scratchpad/pku_contrast.py --root <clone> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

NAME_RE = re.compile(r"^(\d+)_([a-z_]+)_(\d+)$")
CLASS_DIRS = ("Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper")


def read_voc(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append((int(float(b.findtext("xmin"))), int(float(b.findtext("ymin"))),
                      int(float(b.findtext("xmax"))), int(float(b.findtext("ymax"))),
                      (obj.findtext("name") or "").strip()))
    return wh, boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-class", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    # --- 1. XML size vs real JPEG size --------------------------------------
    bad = 0
    checked = 0
    for cls_dir in CLASS_DIRS:
        for ip in sorted((root / "images" / cls_dir).glob("*.jpg"))[:20]:
            wh, _ = read_voc(root / "Annotations" / cls_dir / f"{ip.stem}.xml")
            with Image.open(ip) as im:
                real = im.size
            checked += 1
            if real != wh:
                bad += 1
                if bad <= 5:
                    print(f"  SIZE MISMATCH {ip.name}: xml {wh} vs jpeg {real}")
    print(f"xml size vs jpeg size: {checked - bad}/{checked} agree")

    # template sizes too
    for tp in sorted((root / "PCB_USED").glob("*.JPG")):
        with Image.open(tp) as im:
            print(f"  template {tp.stem}: {im.size[0]}x{im.size[1]}")

    # --- 2. raw contrast inside boxes ---------------------------------------
    templates = {}
    for tp in sorted((root / "PCB_USED").glob("*.JPG")):
        with Image.open(tp) as im:
            templates[tp.stem] = np.asarray(im.convert("RGB"), dtype=np.uint8)

    stats = defaultdict(lambda: defaultdict(list))
    sheets: dict[str, list] = {}
    for cls_dir in CLASS_DIRS:
        imgs = sorted((root / "images" / cls_dir).glob("*.jpg"))
        pick = rng.choice(len(imgs), size=min(args.per_class, len(imgs)), replace=False)
        for i in sorted(pick):
            ip = imgs[int(i)]
            board = NAME_RE.match(ip.stem).group(1)
            temp = templates[board]
            with Image.open(ip) as im:
                test = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if test.shape != temp.shape:
                continue
            d = np.abs(test.astype(np.int16) - temp.astype(np.int16))
            _, boxes = read_voc(root / "Annotations" / cls_dir / f"{ip.stem}.xml")
            for x1, y1, x2, y2, name in boxes:
                box_d = d[y1:y2, x1:x2]
                for ci, c in enumerate("rgb"):
                    stats[name][f"max_{c}"].append(float(box_d[..., ci].max()))
                    stats[name][f"p99_{c}"].append(float(np.percentile(box_d[..., ci], 99)))
                stats[name]["frac_gt30"].append(
                    float((box_d.max(axis=2) > 30).mean()))
                stats[name]["frac_gt10"].append(
                    float((box_d.max(axis=2) > 10).mean()))
            if name not in sheets and boxes:
                x1, y1, x2, y2, name0 = boxes[0]
                pad = 60
                ys = slice(max(0, y1 - pad), y2 + pad)
                xs = slice(max(0, x1 - pad), x2 + pad)
                amp = np.clip(d[ys, xs].astype(np.int32) * 6, 0, 255).astype(np.uint8)
                panel = np.concatenate([temp[ys, xs], test[ys, xs], amp], axis=1)
                sheets[name0] = panel

    print(f"\nraw |test - template| inside annotated boxes "
          f"(median over boxes, {args.per_class} images/class):")
    print(f"{'class':<18}{'max R':>8}{'max G':>8}{'max B':>8}"
          f"{'p99 R':>8}{'% px >30':>10}{'% px >10':>10}")
    for name in sorted(stats):
        s = stats[name]
        print(f"{name:<18}"
              f"{np.median(s['max_r']):>8.0f}{np.median(s['max_g']):>8.0f}"
              f"{np.median(s['max_b']):>8.0f}{np.median(s['p99_r']):>8.0f}"
              f"{100 * np.median(s['frac_gt30']):>10.1f}{100 * np.median(s['frac_gt10']):>10.1f}")

    for name, panel in sheets.items():
        p = out_dir / f"box_{name}.png"
        Image.fromarray(panel).save(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
