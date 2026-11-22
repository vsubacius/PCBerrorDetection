"""PKU-Market-PCB -> DeepPCB-format pairs, so the whole pipeline is reused.

PKU is 693 photographs of 10 bare boards, each with a defect-free template in
``PCB_USED/`` and PASCAL VOC boxes for six defect types. DeepPCB is 1,500
pre-binarized, pre-aligned 640x640 pairs. This module turns the first into the
second: resample, align, binarize, cut 640x640 crops, and write them out in
DeepPCB's on-disk shape. ``build_tiles.py --dataset pku`` then builds tiles
from them with no other change, and ``eval/gate.py``, ``tiling.py``,
``tile_dataset.py`` and the firmware gate all work as they are.

Why a photograph can stand in for DeepPCB's scans at all: on the **green**
channel, copper under green solder mask is brighter than bare laminate, so a
global Otsu recovers a trace map that looks like DeepPCB's. On red it does not
-- see ``data/binarize.py``.

Four things measured on the dataset drive the choices here (full numbers in
``PKU_STEP1_REPORT.md``):

* **Three capture scales.** Pad grids measure 57 px per 2.54 mm on boards
  01/04/05/06, 69 px on 07/08/09 and 103 px on 10/11/12 -- 22.4, 27.2 and
  40.6 px/mm. Every board is resampled to one common scale, per board.
* **The common scale is 22.4 px/mm, not DeepPCB's ~48.** Matching DeepPCB
  would mean upsampling by up to 2.1x, and it would break the tile geometry:
  PKU defects are ~3x larger physically than DeepPCB's (1.96 mm vs 0.65 mm
  median short side), so at 48 px/mm the p90 short side of a `short` reaches
  190 px -- wider than a 160 px tile, which the 70% labelling rule could then
  never mark positive. At 22.4 px/mm a 160 px tile covers 7.1 mm and the
  median defect is 27% of it, close to DeepPCB's own ratio.
* **The pairs are already aligned.** Every sampled pair had a best shift of
  exactly (0, 0). The search runs anyway, because a printed-and-photographed
  board will not be aligned and the same code path serves both.
* **Copper is 0.** DeepPCB's files have copper *black*, so the binarization is
  inverted to match, and a model trained on one dataset sees the same
  convention as a model trained on the other.

Usage::

    python data/pku.py --download                 # clone the dataset mirror
    python data/pku.py --convert                  # write data/PKU-crops/
    python data/pku.py --measure                  # re-measure the pad pitches
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binarize import binarize  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

#: GitHub mirror of PKU-Market-PCB. The official page (robotics.pkusz.edu.cn,
#: Open resources) serves the same data over Baidu Pan, which needs an account.
PKU_URL = "https://github.com/Ironbrotherstyle/PCB-DATASET.git"
PKU_REVISION = "a5769f5"

IMAGE_SIZE = 640

#: PKU class directory and VOC <name> -> DeepPCB class id.
#:
#: Five of the six classes match DeepPCB by name. The sixth does not: DeepPCB's
#: id 6 is pin-hole (a void *in* copper), PKU's is missing_hole (an undrilled
#: pad). They are different defects and are not merged -- id 6 simply means
#: whichever one the active dataset has, and the class names travel with the
#: built dataset so nothing downstream hardcodes it.
VOC_CLASS_IDS: dict[str, int] = {
    "open_circuit": 1,
    "short": 2,
    "mouse_bite": 3,
    "spur": 4,
    "spurious_copper": 5,
    "missing_hole": 6,
}
#: The rest of this block mirrors data/deeppcb.py's interface exactly, so the
#: two modules are interchangeable everywhere downstream.
CLASS_NAMES: dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "missing_hole",
}
NUM_CLASSES = len(CLASS_NAMES)
CLASS_IDS: tuple[int, ...] = tuple(sorted(CLASS_NAMES))
CLASS_ID_TO_CHANNEL: dict[int, int] = {c: i for i, c in enumerate(CLASS_IDS)}

#: Measured pad-grid pitch per board, in pixels per 2.54 mm (0.1 inch).
#:
#: From the autocorrelation of the pad mask, strongest peak in the 30-200 px
#: band -- below 30 px the hatched ground-plane dots (~12 px) dominate.
#: Cross-checked by eye on board 01. Re-measure with ``--measure``.
BOARD_PITCH_PX: dict[str, int] = {
    "01": 57, "04": 57, "05": 57, "06": 57,
    "07": 69, "08": 69, "09": 69,
    "10": 103, "11": 103, "12": 103,
}

#: One header pitch in millimetres.
PITCH_MM = 2.54

#: The scale every board is resampled to, as a pad pitch in pixels. 57 px is
#: 22.4 px/mm, the scale four of the ten boards already have.
TARGET_PITCH_PX = 57

#: Default fold assignment, by board. Test and val span different capture
#: scales so a fold is not also a scale group. The printed demo must use a
#: test board.
DEFAULT_TEST_BOARDS = ("06", "11")
DEFAULT_VAL_BOARDS = ("08",)

#: Keep a defect-free crop for every this-many defect crops. PKU's defect
#: density is far below DeepPCB's, and clean crops are where the gate earns
#: its energy saving, so they are not thrown away.
DEFAULT_CLEAN_RATIO = 1.0

#: A box is kept for a crop when at least this much of its area is inside --
#: the same rule tiling.py uses for tiles.
KEEP_FRAC = 0.7

NAME_RE = re.compile(r"^(\d+)_([a-z_]+)_(\d+)$")


def target_px_per_mm() -> float:
    return TARGET_PITCH_PX / PITCH_MM


def board_px_per_mm(board: str) -> float:
    return BOARD_PITCH_PX[board] / PITCH_MM


def board_scale(board: str) -> float:
    """Resample factor taking this board to the common scale."""
    return TARGET_PITCH_PX / BOARD_PITCH_PX[board]


# --------------------------------------------------------------------------- #
# dataset access
# --------------------------------------------------------------------------- #

def download(dest: str | os.PathLike = "data/PKU-DATASET", force: bool = False) -> Path:
    """Clone the PKU mirror to ``dest`` (no-op if present). ~3.7 GB on disk."""
    dest = Path(dest)
    if force and dest.exists():
        shutil.rmtree(dest)
    if not (dest / "PCB_USED").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", PKU_URL, str(dest)], check=True)
    return dest


def read_voc(path: Path) -> tuple[tuple[int, int], list[tuple[int, int, int, int, int]]]:
    """VOC XML -> ``((width, height), [(x1, y1, x2, y2, class_id), ...])``."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in VOC_CLASS_IDS:
            raise ValueError(f"{path}: unknown class {name!r}")
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (int(float(b.findtext(k)))
                          for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1, y1, x2, y2, VOC_CLASS_IDS[name]))
    return wh, boxes


def raw_index(root: str | os.PathLike) -> list[tuple[str, Path, Path]]:
    """``(board, image, annotation)`` for all 693 non-rotated images."""
    root = Path(root)
    out = []
    for cls_dir in sorted(p.name for p in (root / "images").iterdir() if p.is_dir()):
        for img in sorted((root / "images" / cls_dir).glob("*.jpg")):
            m = NAME_RE.match(img.stem)
            if not m:
                raise ValueError(f"unexpected image name: {img}")
            ann = root / "Annotations" / cls_dir / f"{img.stem}.xml"
            if not ann.exists():
                raise FileNotFoundError(f"no annotation for {img}")
            out.append((m.group(1), img, ann))
    return out


# --------------------------------------------------------------------------- #
# scale measurement
# --------------------------------------------------------------------------- #

def measure_pitch(mask: np.ndarray, lo: int = 30, hi: int = 200,
                  harmonic_frac: float = 0.55) -> int:
    """Dominant pad spacing in px, by autocorrelation of a pad mask.

    Two corrections, both needed on this dataset:

    * the 30 px floor skips the hatched ground-plane dot texture (~12 px),
      which is otherwise the strongest periodicity on these boards;
    * a regular grid also peaks at 2x and 3x its pitch, and on boards 01 and
      05 the harmonic is the *taller* peak. So if a peak near ``p / k`` is
      within ``harmonic_frac`` of the tallest one, the smaller spacing is the
      real pitch.
    """
    m = mask.astype(np.float32)
    m -= m.mean()
    f = np.fft.rfft2(m)
    ac = np.fft.irfft2(f * np.conj(f), s=m.shape)
    ac /= ac.flat[0]

    peaks: dict[int, float] = {}
    for profile in (ac[0, :len(ac[0]) // 2], ac[:len(ac) // 2, 0]):
        seg = profile[lo:min(hi, len(profile))]
        for i in range(1, len(seg) - 1):
            if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
                d = i + lo
                peaks[d] = max(peaks.get(d, -np.inf), float(seg[i]))
    if not peaks:
        return 0

    tallest = max(peaks, key=peaks.get)
    height = peaks[tallest]
    for k in (4, 3, 2):
        candidate = int(round(tallest / k))
        if candidate < lo:
            continue
        nearby = [d for d in peaks
                  if abs(d - candidate) <= 2 and peaks[d] >= harmonic_frac * height]
        if nearby:
            return max(nearby, key=peaks.get)
    return tallest


def measure_all_pitches(root: str | os.PathLike) -> dict[str, int]:
    """Re-measure every board's pad pitch from its template."""
    root = Path(root)
    out = {}
    for tp in sorted((root / "PCB_USED").glob("*.JPG")):
        with Image.open(tp) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # Pads, not traces: red is the channel that isolates them.
        pads, _ = binarize(arr, channel="r")
        out[tp.stem] = measure_pitch(pads)
    return out


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

def resample(arr: np.ndarray, scale: float) -> np.ndarray:
    """Area-average resample of an RGB image. ``scale == 1`` is a no-op."""
    if abs(scale - 1.0) < 1e-9:
        return arr
    h, w = arr.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    with Image.fromarray(arr) as im:
        return np.asarray(im.resize(size, Image.BOX), dtype=np.uint8)


def best_shift(test: np.ndarray, temp: np.ndarray, radius: int = 4,
               window: int = 512) -> tuple[int, int]:
    """Integer ``(dx, dy)`` minimizing XOR over a central window.

    The same criterion the firmware uses to align a camera frame to the stored
    template. On this dataset it always returns (0, 0) -- the test images are
    the template photograph with defects composited in -- but the printed-board
    demo needs it, so the conversion runs through the same path.
    """
    h, w = temp.shape
    side = min(window, h - 2 * radius, w - 2 * radius)
    y0, x0 = (h - side) // 2, (w - side) // 2
    ref = temp[y0:y0 + side, x0:x0 + side]
    best = (0, 0, np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cut = test[y0 + dy:y0 + dy + side, x0 + dx:x0 + dx + side]
            cost = float((cut ^ ref).mean())
            if cost < best[2]:
                best = (dx, dy, cost)
    return best[0], best[1]


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift by whole pixels, filling the exposed border by edge replication."""
    if dx == 0 and dy == 0:
        return img
    return np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)


def crop_origins(length: int, size: int = IMAGE_SIZE) -> list[int]:
    """Non-overlapping crop origins along one axis, last one flush with the edge."""
    if length < size:
        return []
    xs = list(range(0, length - size + 1, size))
    if xs[-1] != length - size:
        xs.append(length - size)
    return xs


def boxes_in_crop(boxes: np.ndarray, x0: int, y0: int,
                  size: int = IMAGE_SIZE, keep_frac: float = KEEP_FRAC) -> np.ndarray:
    """Boxes clipped to a crop, keeping those at least ``keep_frac`` inside."""
    if len(boxes) == 0:
        return np.zeros((0, 5), np.int32)
    x1 = np.clip(boxes[:, 0] - x0, 0, size)
    y1 = np.clip(boxes[:, 1] - y0, 0, size)
    x2 = np.clip(boxes[:, 2] - x0, 0, size)
    y2 = np.clip(boxes[:, 3] - y0, 0, size)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1)
    inside = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    keep = (inside / area) >= keep_frac
    out = np.stack([x1, y1, x2, y2, boxes[:, 4]], axis=1)[keep]
    return out.astype(np.int32)


def write_pair(out_dir: Path, stem: str, test: np.ndarray, temp: np.ndarray,
               boxes: np.ndarray) -> None:
    """One converted pair, in DeepPCB's on-disk shape."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((test * 255).astype(np.uint8)).save(out_dir / f"{stem}_test.png")
    Image.fromarray((temp * 255).astype(np.uint8)).save(out_dir / f"{stem}_temp.png")
    lines = [f"{x1} {y1} {x2} {y2} {c}" for x1, y1, x2, y2, c in boxes]
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def convert(
    root: str | os.PathLike = "data/PKU-DATASET",
    out: str | os.PathLike = "data/PKU-crops",
    test_boards: tuple[str, ...] = DEFAULT_TEST_BOARDS,
    val_boards: tuple[str, ...] = DEFAULT_VAL_BOARDS,
    clean_ratio: float = DEFAULT_CLEAN_RATIO,
    limit: int | None = None,
    seed: int = 0,
) -> Path:
    """Convert the dataset. Returns the output directory."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    samples = raw_index(root)
    if limit:
        samples = samples[:limit]

    # Templates: resample, binarize once per board, and keep the threshold so
    # every test image of that board is thresholded at the same level.
    templates: dict[str, tuple[np.ndarray, int]] = {}
    for board in sorted({b for b, _, _ in samples}):
        with Image.open(root / "PCB_USED" / f"{board}.JPG") as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        arr = resample(arr, board_scale(board))
        mask, threshold = binarize(arr, channel="g", invert=True)
        templates[board] = (mask, threshold)
        print(f"  template {board}: {arr.shape[1]}x{arr.shape[0]} after "
              f"{board_scale(board):.3f}x, otsu(g)={threshold}, copper "
              f"{mask.mean():.1%}")

    index_rows: list[dict] = []
    n_boxes_in = n_boxes_kept = 0
    n_clean_available = n_clean_kept = 0
    shifts: list[tuple[int, int]] = []

    for i, (board, img_path, ann_path) in enumerate(samples):
        scale = board_scale(board)
        temp_mask, threshold = templates[board]

        with Image.open(img_path) as im:
            test_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        _, raw_boxes = read_voc(ann_path)
        test_arr = resample(test_arr, scale)
        test_mask, _ = binarize(test_arr, channel="g", invert=True,
                                threshold=threshold)

        if test_mask.shape != temp_mask.shape:
            print(f"  skipping {img_path.name}: {test_mask.shape} vs template "
                  f"{temp_mask.shape}")
            continue

        dx, dy = best_shift(test_mask, temp_mask)
        shifts.append((dx, dy))
        test_mask = shift_image(test_mask, dx, dy)

        boxes = np.array([[round(x1 * scale) - dx, round(y1 * scale) - dy,
                           round(x2 * scale) - dx, round(y2 * scale) - dy, c]
                          for x1, y1, x2, y2, c in raw_boxes], dtype=np.int32)
        n_boxes_in += len(boxes)

        h, w = temp_mask.shape
        fold = ("test" if board in test_boards
                else "val" if board in val_boards else "train")
        board_dir = out / f"board{board}"

        grid = [(x0, y0) for y0 in crop_origins(h) for x0 in crop_origins(w)]
        defect_origins = [o for o in grid if len(boxes_in_crop(boxes, *o))]
        clean_origins = [o for o in grid if not len(boxes_in_crop(boxes, *o))]

        # A defect that no grid crop holds to 70% -- one straddling a crop
        # boundary -- gets a crop centred on it, so the grid never silently
        # drops an annotation.
        rescued: list[tuple[int, int]] = []
        for box in boxes:
            if any(len(boxes_in_crop(box[None, :], *o)) for o in defect_origins):
                continue
            if w < IMAGE_SIZE or h < IMAGE_SIZE:
                continue
            cx, cy = (int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2
            origin = (int(np.clip(cx - IMAGE_SIZE // 2, 0, w - IMAGE_SIZE)),
                      int(np.clip(cy - IMAGE_SIZE // 2, 0, h - IMAGE_SIZE)))
            if origin not in defect_origins and origin not in rescued:
                rescued.append(origin)

        n_clean_available += len(clean_origins)
        n_keep_clean = int(round(clean_ratio * (len(defect_origins) + len(rescued))))
        if n_keep_clean and clean_origins:
            pick = rng.choice(len(clean_origins),
                              size=min(n_keep_clean, len(clean_origins)), replace=False)
            chosen_clean = [clean_origins[int(j)] for j in sorted(pick)]
        else:
            chosen_clean = []
        n_clean_kept += len(chosen_clean)

        chosen = defect_origins + rescued + chosen_clean
        # Count each annotation once, not once per crop: the flush-edge crops
        # overlap, so a box can land in two of them.
        n_boxes_kept += sum(
            any(len(boxes_in_crop(box[None, :], *o)) for o in chosen) for box in boxes)

        for x0, y0 in chosen:
            stem = f"{img_path.stem}_{x0:04d}_{y0:04d}"
            local = boxes_in_crop(boxes, x0, y0)
            write_pair(board_dir, stem,
                       test_mask[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE],
                       temp_mask[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE],
                       local)
            index_rows.append({
                "stem": f"board{board}/{stem}",
                "board": board,
                "source": img_path.name,
                "crop": [x0, y0],
                "fold": fold,
                "n_boxes": int(len(local)),
            })

        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"  {i + 1}/{len(samples)} images, {len(index_rows)} crops",
                  flush=True)

    meta = {
        "dataset": "PKU-Market-PCB",
        "source_url": PKU_URL,
        "source_revision": PKU_REVISION,
        "image_size": IMAGE_SIZE,
        "target_pitch_px": TARGET_PITCH_PX,
        "target_px_per_mm": target_px_per_mm(),
        "board_pitch_px": BOARD_PITCH_PX,
        "channel": "g",
        "invert": True,
        "keep_frac": KEEP_FRAC,
        "clean_ratio": clean_ratio,
        "class_ids": VOC_CLASS_IDS,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "test_boards": list(test_boards),
        "val_boards": list(val_boards),
        "seed": seed,
    }
    (out / "index.json").write_text(json.dumps({"meta": meta, "samples": index_rows},
                                               indent=1))

    folds = {f: sum(r["fold"] == f for r in index_rows) for f in ("train", "val", "test")}
    print(f"\nwrote {len(index_rows)} crops to {out}")
    print(f"  folds: " + ", ".join(f"{k} {v}" for k, v in folds.items()))
    print(f"  boards: train {sorted({r['board'] for r in index_rows if r['fold'] == 'train'})}, "
          f"val {sorted(val_boards)}, test {sorted(test_boards)}")
    print(f"  boxes: {n_boxes_in} in, {n_boxes_kept} kept "
          f"({n_boxes_kept / max(1, n_boxes_in):.1%})")
    print(f"  clean crops: {n_clean_kept} kept of {n_clean_available} available")
    if shifts:
        mags = [max(abs(dx), abs(dy)) for dx, dy in shifts]
        print(f"  alignment: median |shift| {np.median(mags):.1f} px, max {max(mags)} px")
    return out


# --------------------------------------------------------------------------- #
# the deeppcb.Sample interface, over the converted crops
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Sample:
    """One converted pair. Same interface as ``deeppcb.Sample``."""

    stem: str
    test_path: Path
    temp_path: Path
    ann_path: Path
    split: str
    board: str

    @property
    def name(self) -> str:
        return Path(self.stem).name

    @property
    def group(self) -> str:
        """The physical board -- what a group-level split must not straddle."""
        return f"board{self.board}"

    def boxes(self) -> np.ndarray:
        return read_annotations(self.ann_path)

    def load(self, binarize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``(test, template)`` as ``(640, 640) uint8``, 0/1 when binarize."""
        test = load_png(self.test_path)
        temp = load_png(self.temp_path)
        if binarize:
            return test, temp
        return (test * 255).astype(np.uint8), (temp * 255).astype(np.uint8)


def load_png(path: str | os.PathLike) -> np.ndarray:
    """Load a converted crop as ``{0, 1} uint8``."""
    with Image.open(path) as im:
        return (np.asarray(im.convert("L"), dtype=np.uint8) > 127).astype(np.uint8)


def read_annotations(path: str | os.PathLike) -> np.ndarray:
    """``(n, 5) int32`` of ``[x1, y1, x2, y2, class_id]``, DeepPCB's format."""
    rows = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {parts!r}")
        x1, y1, x2, y2, cls = (int(float(p)) for p in parts)
        if cls not in CLASS_NAMES:
            raise ValueError(f"{path}:{lineno}: unknown class id {cls}")
        if not (0 <= x1 < x2 <= IMAGE_SIZE and 0 <= y1 < y2 <= IMAGE_SIZE):
            raise ValueError(f"{path}:{lineno}: box out of range: {parts!r}")
        rows.append((x1, y1, x2, y2, cls))
    if not rows:
        return np.zeros((0, 5), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def index(root: str | os.PathLike = "data/PKU-crops", split: str | None = None,
          check: bool = True) -> list[Sample]:
    """List converted samples. ``split`` is ``train`` / ``val`` / ``test``.

    Unlike DeepPCB there is no official split to respect: the folds are the
    board-level split chosen at conversion time and recorded in index.json.
    """
    root = Path(root)
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{index_path} not found -- run `python data/pku.py --convert` first.")
    blob = json.loads(index_path.read_text())

    samples = []
    for row in blob["samples"]:
        if split is not None and row["fold"] != split:
            continue
        stem = row["stem"]
        samples.append(Sample(
            stem=stem,
            test_path=root / f"{stem}_test.png",
            temp_path=root / f"{stem}_temp.png",
            ann_path=root / f"{stem}.txt",
            split=row["fold"],
            board=row["board"],
        ))

    if check:
        missing = [str(p) for s in samples
                   for p in (s.test_path, s.temp_path, s.ann_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} converted file(s) missing, e.g. {missing[:3]}")
    return samples


def meta(root: str | os.PathLike = "data/PKU-crops") -> dict:
    return json.loads((Path(root) / "index.json").read_text())["meta"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/PKU-DATASET")
    ap.add_argument("--out", default="data/PKU-crops")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--measure", action="store_true",
                    help="re-measure the pad pitch of every board and exit")
    ap.add_argument("--test-boards", default=",".join(DEFAULT_TEST_BOARDS))
    ap.add_argument("--val-boards", default=",".join(DEFAULT_VAL_BOARDS))
    ap.add_argument("--clean-ratio", type=float, default=DEFAULT_CLEAN_RATIO)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.download:
        download(args.root)

    if args.measure:
        measured = measure_all_pitches(args.root)
        print(f"{'board':<8}{'measured':>10}{'recorded':>10}{'px/mm':>9}{'scale':>8}")
        for board, pitch in measured.items():
            recorded = BOARD_PITCH_PX.get(board)
            print(f"{board:<8}{pitch:>10}{recorded if recorded else '-':>10}"
                  f"{pitch / PITCH_MM:>9.1f}{TARGET_PITCH_PX / pitch:>8.3f}")
        return

    if args.convert:
        convert(
            root=args.root,
            out=args.out,
            test_boards=tuple(args.test_boards.split(",")),
            val_boards=tuple(b for b in args.val_boards.split(",") if b),
            clean_ratio=args.clean_ratio,
            limit=args.limit,
            seed=args.seed,
        )
    elif not args.download:
        ap.error("nothing to do: pass --download, --convert or --measure")


if __name__ == "__main__":
    main()
