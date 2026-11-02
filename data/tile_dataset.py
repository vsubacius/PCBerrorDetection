"""Load the built tile dataset, cut tiles on the fly, and augment them.

The ``.npz`` written by ``build_tiles.py`` stores whole bit-packed images plus
per-tile labels. :class:`TileData` unpacks one fold into memory and
:meth:`TileData.tiles` cuts the tiles for an image, applying the augmentations
and building the 1- or 2-channel input the ablation asks for:

=========  ==================================================================
 ``"A"``    test tile only (1 channel) -- no reference, the weakest baseline
 ``"B"``    test + template (2 channels) -- the model learns the comparison
 ``"C"``    test + XOR(test, template) (2 channels) -- the difference is
            handed to the model, which is also what the classical gate sees
=========  ==================================================================

Augmentations (all applied *identically* to test and template so the pair
stays aligned, except the misalignment shift, whose whole point is to break
alignment slightly):

* flips and 90-degree rotations
* a small random x/y shift of the template against the test image, simulating
  the residual registration error a real camera pipeline leaves behind
* random erosion / dilation, simulating binarization-threshold drift

Augmentation happens at the *image* level before cutting, so a defect box is
transformed with it and tile labels stay correct.

:class:`TorchTileDataset` wraps this as a ``torch.utils.data.Dataset`` yielding
one tile at a time; torch is imported lazily so the module works without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deeppcb import IMAGE_SIZE, NUM_CLASSES
from tiling import TileGrid, label_tiles

REPRESENTATIONS = ("A", "B", "C")


@dataclass
class AugmentConfig:
    """Augmentation switches. All off by default (use for val/test)."""

    flip: bool = False
    rot90: bool = False
    max_shift: int = 0           # px of template-vs-test misalignment
    morph_prob: float = 0.0      # chance of an erosion or dilation
    morph_radius: int = 1        # 1 -> 3x3 structuring element

    @classmethod
    def train_default(cls) -> "AugmentConfig":
        return cls(flip=True, rot90=True, max_shift=3, morph_prob=0.3, morph_radius=1)


def _unpack(packed: np.ndarray) -> np.ndarray:
    """``(..., 640, 80) uint8`` bit-packed -> ``(..., 640, 640) uint8`` of 0/1."""
    return np.unpackbits(packed, axis=-1, count=IMAGE_SIZE)


def binary_morph(img: np.ndarray, dilate: bool, radius: int) -> np.ndarray:
    """Binary dilation/erosion with a square element, via shifted maxima.

    Hand-rolled so the pipeline needs no scipy/cv2 (keeps Colab setup thin).
    """
    out = img
    k = 2 * radius + 1
    pad = np.pad(img, radius, mode="edge")
    stack = np.stack(
        [pad[dy : dy + img.shape[0], dx : dx + img.shape[1]]
         for dy in range(k) for dx in range(k)],
        axis=0,
    )
    out = stack.max(axis=0) if dilate else stack.min(axis=0)
    return out.astype(np.uint8)


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an image, padding with the edge value (never with fake black)."""
    if dx == 0 and dy == 0:
        return img
    pad = max(abs(dx), abs(dy))
    p = np.pad(img, pad, mode="edge")
    y0, x0 = pad - dy, pad - dx
    return p[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]]


def _flip_rot_boxes(boxes: np.ndarray, flip_x: bool, flip_y: bool, k: int,
                    size: int = IMAGE_SIZE) -> np.ndarray:
    """Apply the same flips/rotations to boxes that were applied to the image."""
    if boxes.size == 0:
        return boxes
    b = boxes.astype(np.int32).copy()
    x1, y1, x2, y2 = b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy()
    if flip_x:
        x1, x2 = size - x2, size - x1
    if flip_y:
        y1, y2 = size - y2, size - y1
    for _ in range(k % 4):
        # np.rot90 (counter-clockwise): (x, y) -> (y, size - x)
        x1, y1, x2, y2 = y1, size - x2, y2, size - x1
    b[:, 0], b[:, 1], b[:, 2], b[:, 3] = x1, y1, x2, y2
    return b


@dataclass
class TileData:
    """One fold of the built dataset, unpacked and ready to serve tiles."""

    test: np.ndarray            # (N, 640, 640) uint8 0/1
    temp: np.ndarray            # (N, 640, 640) uint8 0/1
    y: np.ndarray               # (N, n_tiles, 6) float32
    ignore: np.ndarray          # (N, n_tiles, 6) bool
    boxes: list[np.ndarray]     # per image, (n_i, 5) int32
    names: np.ndarray
    groups: np.ndarray
    grid: TileGrid
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.test)

    @property
    def n_tiles(self) -> int:
        return self.grid.n_tiles

    @classmethod
    def load(cls, path: str | Path, fold: str | None = None,
             grid: TileGrid | None = None) -> "TileData":
        """Read an ``.npz`` and keep one fold (``train``/``val``/``test``).

        Pass ``grid`` to re-tile at a different size than the file was built
        with: the labels are recomputed from the stored boxes, so the tile-size
        ablation needs no rebuild.
        """
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            folds = z["fold"]
            sel = np.ones(len(folds), dtype=bool) if fold is None else (folds == fold)
            if not sel.any():
                raise ValueError(f"fold {fold!r} not in {sorted(set(folds.tolist()))}")
            idx = np.flatnonzero(sel)

            test = _unpack(z["images_test"][idx])
            temp = _unpack(z["images_temp"][idx])
            boxes_all, box_image = z["boxes"], z["box_image"]
            remap = {old: new for new, old in enumerate(idx)}
            per_image: list[np.ndarray] = [np.zeros((0, 5), np.int32) for _ in idx]
            for old, new in remap.items():
                per_image[new] = boxes_all[box_image == old]

            file_grid = TileGrid(meta["tile"], meta["stride"], meta["image_size"])
            use_grid = grid or file_grid
            if use_grid == file_grid:
                y = z["y"][idx].astype(np.float32)
                ignore = z["ignore"][idx].astype(bool)
            else:
                ys, igs = zip(*(
                    label_tiles(b, use_grid, meta["pos_thresh"], meta["ignore_thresh"])
                    for b in per_image
                ))
                y, ignore = np.stack(ys), np.stack(igs)

            return cls(
                test=test, temp=temp, y=y, ignore=ignore, boxes=per_image,
                names=z["name"][idx], groups=z["group"][idx],
                grid=use_grid, meta=meta,
            )

    def tiles(
        self,
        i: int,
        representation: str = "B",
        aug: AugmentConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tiles, targets and ignore masks for image ``i``.

        Returns ``(x, y, ignore)`` with ``x`` of shape
        ``(n_tiles, C, tile, tile) float32`` in ``{0, 1}``, ``C`` 1 or 2.
        """
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        rng = rng or np.random.default_rng()
        test, temp = self.test[i], self.temp[i]
        y, ignore = self.y[i], self.ignore[i]

        if aug is not None:
            test, temp, y, ignore = self._augment(test, temp, self.boxes[i], aug, rng)

        xor = None
        if representation == "C":
            xor = (test ^ temp).astype(np.uint8)

        layers = {"A": [test], "B": [test, temp], "C": [test, xor]}[representation]
        cut = np.stack([self.grid.cut(layer) for layer in layers], axis=1)
        return cut.astype(np.float32), y, ignore

    def _augment(self, test, temp, boxes, aug: AugmentConfig, rng):
        flip_x = bool(aug.flip and rng.integers(2))
        flip_y = bool(aug.flip and rng.integers(2))
        k = int(rng.integers(4)) if aug.rot90 else 0

        if flip_x:
            test, temp = test[:, ::-1], temp[:, ::-1]
        if flip_y:
            test, temp = test[::-1], temp[::-1]
        if k:
            test, temp = np.rot90(test, k), np.rot90(temp, k)
        test, temp = np.ascontiguousarray(test), np.ascontiguousarray(temp)

        # Misalignment: shift the template only, so the pair stops matching the
        # way a real registration error would. Labels are unaffected -- the
        # defects live in the test image.
        if aug.max_shift:
            dx = int(rng.integers(-aug.max_shift, aug.max_shift + 1))
            dy = int(rng.integers(-aug.max_shift, aug.max_shift + 1))
            temp = shift_image(temp, dx, dy)

        # Threshold drift: same morphology on both images (one binarizer saw
        # both), so the gate's XOR stays meaningful.
        if aug.morph_prob and rng.random() < aug.morph_prob:
            dilate = bool(rng.integers(2))
            test = binary_morph(test, dilate, aug.morph_radius)
            temp = binary_morph(temp, dilate, aug.morph_radius)

        boxes_t = _flip_rot_boxes(boxes, flip_x, flip_y, k)
        y, ignore = label_tiles(
            boxes_t, self.grid,
            self.meta.get("pos_thresh", 0.7), self.meta.get("ignore_thresh", 0.0),
        )
        return test, temp, y, ignore

    def flat_tiles(self, representation: str = "B", aug: AugmentConfig | None = None,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every tile of every image, stacked. Handy for PTQ calibration sets.

        Beware the memory: 1,000 images x 25 tiles x 2 x 160x160 float32 is
        ~5 GB. Slice the fold or use :class:`TorchTileDataset` for training.
        """
        rng = np.random.default_rng(seed)
        xs, ys, igs = [], [], []
        for i in range(len(self)):
            x, y, ig = self.tiles(i, representation, aug, rng)
            xs.append(x), ys.append(y), igs.append(ig)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(igs)

    def class_pos_weights(self) -> np.ndarray:
        """Inverse-frequency positive weights per class, for the weighted BCE.

        ``w_c = n_neg_c / n_pos_c`` over non-ignored tiles -- the
        ``inverse class-frequency`` factor in the defect-aware KD loss.
        """
        valid = ~self.ignore
        pos = (self.y > 0) & valid
        n_pos = pos.reshape(-1, NUM_CLASSES).sum(0).astype(np.float64)
        n_val = valid.reshape(-1, NUM_CLASSES).sum(0).astype(np.float64)
        return np.where(n_pos > 0, (n_val - n_pos) / np.maximum(n_pos, 1), 1.0)


class TorchTileDataset:
    """``torch.utils.data.Dataset`` over individual tiles.

    Indexing is ``(image, tile)`` flattened, so a shuffled DataLoader still
    mixes tiles from different images. Augmentation is re-rolled per image
    access, which is what we want: every epoch sees a different transform.
    """

    def __init__(self, data: TileData, representation: str = "B",
                 aug: AugmentConfig | None = None, seed: int = 0,
                 drop_clean_frac: float = 0.0):
        import torch  # noqa: F401  (fail early with a clear message)

        self.data = data
        self.representation = representation
        self.aug = aug
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int]] = [
            (i, t) for i in range(len(data)) for t in range(data.n_tiles)
        ]
        if drop_clean_frac > 0:
            keep = []
            for i, t in self.index:
                clean = not (data.y[i, t] > 0).any() and not data.ignore[i, t].any()
                if clean and self.rng.random() < drop_clean_frac:
                    continue
                keep.append((i, t))
            self.index = keep

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        import torch

        i, t = self.index[k]
        x, y, ig = self.data.tiles(i, self.representation, self.aug, self.rng)
        return (
            torch.from_numpy(np.ascontiguousarray(x[t])),
            torch.from_numpy(y[t]),
            torch.from_numpy(ig[t].astype(np.float32)),
        )
