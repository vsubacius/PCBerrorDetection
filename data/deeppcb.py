"""DeepPCB dataset access: download, index, and load image pairs.

DeepPCB (https://github.com/tangsanli5201/DeepPCB) ships 1,500 aligned
(template, test) image pairs of binarized bare-PCB photos, 640x640, with
axis-aligned defect boxes on the test image.

On-disk layout inside the cloned repo::

    PCBData/
      trainval.txt                      # 1000 lines: "<stem>.jpg <ann>.txt"
      test.txt                          #  500 lines, same format
      group<ID>/
        <ID>/<name>_test.jpg            # defective image
        <ID>/<name>_temp.jpg            # defect-free template
        <ID>_not/<name>.txt             # annotations, "x1 y1 x2 y2 type"

Quirks this module handles, all verified against the real clone:

* The split lists name ``<group>/<id>/<name>.jpg``, which is *not* a file that
  exists -- the real files have ``_test`` / ``_temp`` appended to the stem.
* One template, ``group90100/90100/90100034_temp.jpg``, has no matching test
  image and no annotation. It is not in either split list and is skipped.
* At least one JPEG is stored as RGB rather than grayscale, so every image is
  converted to mode ``L`` before use.
* The images were binarized *before* JPEG compression, so ~1% of pixels come
  back mid-gray. :func:`load_binary` re-thresholds to a true {0,1} image, which
  matters because the Stage-1 gate counts XOR pixels.
* Annotation coordinates are 1-based-ish inclusive pixel corners in the
  original release; we treat them as a half-open ``[x1, x2) x [y1, y2)`` box,
  which is what the tiling code expects. All 10,013 boxes are in range.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEEPPCB_URL = "https://github.com/tangsanli5201/DeepPCB.git"

#: Defect type id -> short name. Ids come from the dataset README; id 0 is
#: "background" and never appears in annotation files.
CLASS_NAMES: dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",       # "spurious copper"
    6: "pin-hole",
}
NUM_CLASSES = len(CLASS_NAMES)
#: Class ids in the fixed channel order used by every model head in this repo.
CLASS_IDS: tuple[int, ...] = tuple(sorted(CLASS_NAMES))
#: Channel index (0..5) for a dataset class id (1..6).
CLASS_ID_TO_CHANNEL: dict[int, int] = {c: i for i, c in enumerate(CLASS_IDS)}

IMAGE_SIZE = 640
#: The one template with no test-image partner (see module docstring).
KNOWN_ORPHAN_TEMPLATES = ("group90100/90100/90100034_temp.jpg",)


@dataclass(frozen=True)
class Sample:
    """One aligned (template, test) pair plus its defect boxes."""

    stem: str          # e.g. "group20085/20085/20085000"
    test_path: Path
    temp_path: Path
    ann_path: Path
    split: str         # "trainval" or "test" (the dataset's official split)

    @property
    def name(self) -> str:
        """Short unique id, e.g. ``20085000``."""
        return Path(self.stem).name

    @property
    def group(self) -> str:
        """PCB group, e.g. ``group20085``. Images in a group share a board."""
        return self.stem.split("/", 1)[0]

    def boxes(self) -> np.ndarray:
        """``(n, 5) int32`` array of ``[x1, y1, x2, y2, class_id]``."""
        return read_annotations(self.ann_path)

    def load(self, binarize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(test, template)`` as ``(640, 640) uint8`` arrays.

        With ``binarize=True`` (the default) values are 0/1; otherwise 0..255.
        """
        loader = load_binary if binarize else load_gray
        return loader(self.test_path), loader(self.temp_path)


def download(dest: str | os.PathLike = "data/DeepPCB", force: bool = False) -> Path:
    """Shallow-clone DeepPCB to ``dest`` (no-op if already present).

    ~221 MB on disk. Returns the path to the clone's ``PCBData`` directory.
    """
    dest = Path(dest)
    if force and dest.exists():
        shutil.rmtree(dest)
    if not (dest / "PCBData").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", DEEPPCB_URL, str(dest)],
            check=True,
        )
    return dest / "PCBData"


def _read_split_list(pcb_data: Path, split: str) -> list[Sample]:
    list_path = pcb_data / f"{'trainval' if split == 'trainval' else 'test'}.txt"
    samples: list[Sample] = []
    for raw in list_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        img_rel, ann_rel = line.split()
        stem = img_rel[: -len(".jpg")] if img_rel.endswith(".jpg") else img_rel
        samples.append(
            Sample(
                stem=stem,
                test_path=pcb_data / f"{stem}_test.jpg",
                temp_path=pcb_data / f"{stem}_temp.jpg",
                ann_path=pcb_data / ann_rel,
                split=split,
            )
        )
    return samples


def index(root: str | os.PathLike = "data/DeepPCB", split: str | None = None,
          check: bool = True) -> list[Sample]:
    """List dataset samples.

    Args:
        root: the DeepPCB clone, or its ``PCBData`` subdirectory.
        split: ``"trainval"``, ``"test"``, or ``None`` for both.
        check: verify every referenced file exists (cheap, ~3k ``stat`` calls).
    """
    root = Path(root)
    pcb_data = root if root.name == "PCBData" else root / "PCBData"
    if not pcb_data.is_dir():
        raise FileNotFoundError(
            f"{pcb_data} not found -- run deeppcb.download() first."
        )

    splits = ("trainval", "test") if split is None else (split,)
    for s in splits:
        if s not in ("trainval", "test"):
            raise ValueError(f"split must be 'trainval', 'test' or None, got {s!r}")

    samples = [s for name in splits for s in _read_split_list(pcb_data, name)]

    if check:
        missing = [
            str(p)
            for s in samples
            for p in (s.test_path, s.temp_path, s.ann_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} referenced file(s) missing, e.g. {missing[:3]}"
            )
        stems = [s.stem for s in samples]
        if len(set(stems)) != len(stems):
            raise ValueError("duplicate stems in the official split lists")
    return samples


def read_annotations(path: str | os.PathLike) -> np.ndarray:
    """Parse one annotation file into ``(n, 5) int32``: x1, y1, x2, y2, class_id.

    Boxes are returned as half-open ``[x1, x2) x [y1, y2)``.
    """
    rows: list[tuple[int, int, int, int, int]] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip().replace(",", " ")
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


def load_gray(path: str | os.PathLike) -> np.ndarray:
    """Load an image as ``(H, W) uint8`` grayscale (handles the stray RGB file)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_binary(path: str | os.PathLike, threshold: int = 127) -> np.ndarray:
    """Load an image and re-threshold it to a true binary ``{0, 1} uint8`` array.

    DeepPCB images are binarized but JPEG-compressed, so ~1% of pixels arrive
    mid-gray. Copper is white (1) and substrate black (0).
    """
    return (load_gray(path) > threshold).astype(np.uint8)


def class_name(class_id: int) -> str:
    return CLASS_NAMES[class_id]
