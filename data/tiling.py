"""Tile grid geometry and multi-label tile labelling for the gated cascade.

The cascade works on fixed tiles cut from the 640x640 pair, not on whole
images, so every image becomes ``grid.n_tiles`` training examples with a
6-channel multi-label target plus a per-class *ignore* mask.

Labelling rule (per tile, per class ``c``), driven by how much of a defect box
falls inside the tile:

* ``frac >= pos_thresh`` for some box of class ``c``  -> target 1
* else ``frac > ignore_thresh`` for some box of class ``c`` -> **ignore**
  (masked out of the loss: the tile shows a fragment too small to name)
* else -> target 0

``frac`` is *the fraction of the box's area* that lies inside the tile, so a
whole defect inside a big tile scores 1.0 regardless of tile size. A tile with
no target 1 and nothing ignored is a *clean* tile, which is what the Stage-1
gate should let through without running Stage 2.

Coordinates are half-open everywhere: a tile at origin ``(ox, oy)`` covers
``[ox, ox + tile) x [oy, oy + tile)``, matching ``img[oy:oy+tile, ox:ox+tile]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deeppcb import CLASS_IDS, IMAGE_SIZE, NUM_CLASSES

#: Default labelling thresholds (see module docstring).
POS_THRESH = 0.7
IGNORE_THRESH = 0.0


@dataclass(frozen=True)
class TileGrid:
    """A regular, possibly overlapping square tile grid over a square image.

    Args:
        tile: tile side in pixels.
        stride: step between tile origins; ``tile - stride`` is the overlap.
        image_size: side of the source image.

    The last tile is always flush with the image edge. If ``stride`` does not
    divide the remainder evenly, the final step is short rather than letting a
    tile hang off the edge (no padding, so the gate never sees fake black).
    """

    tile: int = 160
    stride: int = 120
    image_size: int = IMAGE_SIZE

    def __post_init__(self) -> None:
        if not 0 < self.tile <= self.image_size:
            raise ValueError(f"tile must be in (0, {self.image_size}], got {self.tile}")
        if not 0 < self.stride <= self.tile:
            raise ValueError(f"stride must be in (0, tile], got {self.stride}")

    @property
    def overlap(self) -> int:
        """Pixels shared between neighbouring tiles."""
        return self.tile - self.stride

    @property
    def origins_1d(self) -> np.ndarray:
        """Tile origins along one axis, last one flush with the edge."""
        last = self.image_size - self.tile
        xs = list(range(0, last + 1, self.stride))
        if xs[-1] != last:
            xs.append(last)
        return np.asarray(xs, dtype=np.int32)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the tile grid."""
        n = len(self.origins_1d)
        return (n, n)

    @property
    def n_tiles(self) -> int:
        rows, cols = self.grid_shape
        return rows * cols

    def tiles(self) -> np.ndarray:
        """``(n_tiles, 4) int32`` of ``[x1, y1, x2, y2]``, row-major (y outer)."""
        xs = self.origins_1d
        yy, xx = np.meshgrid(xs, xs, indexing="ij")
        x1 = xx.ravel()
        y1 = yy.ravel()
        return np.stack([x1, y1, x1 + self.tile, y1 + self.tile], axis=1).astype(np.int32)

    def tile_index(self, row: int, col: int) -> int:
        return row * self.grid_shape[1] + col

    def cut(self, image: np.ndarray) -> np.ndarray:
        """Cut an ``(H, W)`` or ``(H, W, C)`` image into ``(n_tiles, tile, tile[, C])``."""
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise ValueError(
                f"expected a {self.image_size}x{self.image_size} image, "
                f"got {image.shape[:2]}"
            )
        t = self.tile
        out = [image[y : y + t, x : x + t] for x, y, _, _ in self.tiles()]
        return np.stack(out, axis=0)

    @property
    def pixel_cost(self) -> int:
        """Total pixels Stage 2 would see per board with the gate disabled."""
        return self.n_tiles * self.tile * self.tile

    def describe(self) -> str:
        rows, cols = self.grid_shape
        return (
            f"tile={self.tile} stride={self.stride} overlap={self.overlap} "
            f"grid={rows}x{cols} ({self.n_tiles} tiles) "
            f"pixel_cost={self.pixel_cost / self.image_size**2:.2f}x image"
        )


def box_overlap_fracs(boxes: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Fraction of each box's area that lies inside each tile.

    Args:
        boxes: ``(n_boxes, >=4)``, half-open ``[x1, y1, x2, y2, ...]``.
        tiles: ``(n_tiles, 4)``, half-open.

    Returns:
        ``(n_boxes, n_tiles) float32`` in ``[0, 1]``.
    """
    boxes = np.asarray(boxes)
    tiles = np.asarray(tiles)
    if boxes.size == 0:
        return np.zeros((0, len(tiles)), dtype=np.float32)

    bx1, by1, bx2, by2 = (boxes[:, i].astype(np.float64)[:, None] for i in range(4))
    tx1, ty1, tx2, ty2 = (tiles[:, i].astype(np.float64)[None, :] for i in range(4))

    iw = np.clip(np.minimum(bx2, tx2) - np.maximum(bx1, tx1), 0, None)
    ih = np.clip(np.minimum(by2, ty2) - np.maximum(by1, ty1), 0, None)
    area = np.clip((bx2 - bx1) * (by2 - by1), 1e-9, None)
    return ((iw * ih) / area).astype(np.float32)


def label_tiles(
    boxes: np.ndarray,
    grid: TileGrid,
    pos_thresh: float = POS_THRESH,
    ignore_thresh: float = IGNORE_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-label targets and ignore masks for every tile of one image.

    Args:
        boxes: ``(n, 5)`` of ``[x1, y1, x2, y2, class_id]`` (class ids 1..6).
        grid: the tile grid.
        pos_thresh: box-area fraction at or above which the class is positive.
        ignore_thresh: box-area fraction above which a sub-threshold fragment
            makes the class *ignored* rather than negative.

    Returns:
        ``(y, ignore)``, both ``(n_tiles, 6)``; ``y`` is float32 0/1 and
        ``ignore`` is bool. ``y`` is 0 wherever ``ignore`` is True, so a
        masked loss can use ``y`` directly.
    """
    if not 0.0 < pos_thresh <= 1.0:
        raise ValueError(f"pos_thresh must be in (0, 1], got {pos_thresh}")
    if not 0.0 <= ignore_thresh < pos_thresh:
        raise ValueError("need 0 <= ignore_thresh < pos_thresh")

    tiles = grid.tiles()
    y = np.zeros((len(tiles), NUM_CLASSES), dtype=np.float32)
    ignore = np.zeros((len(tiles), NUM_CLASSES), dtype=bool)

    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return y, ignore

    fracs = box_overlap_fracs(boxes, tiles)            # (n_boxes, n_tiles)
    for channel, class_id in enumerate(CLASS_IDS):
        rows = boxes[:, 4] == class_id
        if not rows.any():
            continue
        f = fracs[rows]                                 # (n_class_boxes, n_tiles)
        pos = (f >= pos_thresh).any(axis=0)
        frag = (f > ignore_thresh).any(axis=0) & ~pos
        y[pos, channel] = 1.0
        ignore[frag, channel] = True
    return y, ignore


def gate_labels(y: np.ndarray, ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-class tile labels into Stage-1 gate targets.

    A tile is gate-positive if any class is positive, gate-ignored if it only
    holds fragments, and gate-negative (clean) otherwise.
    """
    gate_y = (y > 0).any(axis=1).astype(np.float32)
    gate_ignore = ignore.any(axis=1) & (gate_y == 0)
    return gate_y, gate_ignore


def is_clean(y: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """``(n_tiles,) bool``: tiles with no positive class and no fragment."""
    return ~(y > 0).any(axis=1) & ~ignore.any(axis=1)


def best_tile_per_box(boxes: np.ndarray, grid: TileGrid) -> tuple[np.ndarray, np.ndarray]:
    """For each box, the tile holding most of it, and that fraction.

    Used by the defect-hit evaluation metric: a defect is *reachable* only if
    some tile holds enough of it to be labelled positive there.
    """
    fracs = box_overlap_fracs(boxes, grid.tiles())
    if fracs.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    return fracs.argmax(axis=1).astype(np.int32), fracs.max(axis=1)
