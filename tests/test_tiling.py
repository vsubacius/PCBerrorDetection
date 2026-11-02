"""Unit tests for the tile grid and the tile-labelling rule.

Run from the repo root with:  python -m pytest tests -q
(or `python tests/test_tiling.py` for a dependency-free smoke run).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from deeppcb import CLASS_ID_TO_CHANNEL, NUM_CLASSES  # noqa: E402
from tiling import (  # noqa: E402
    TileGrid,
    best_tile_per_box,
    box_overlap_fracs,
    gate_labels,
    is_clean,
    label_tiles,
)


def test_grid_geometry_default():
    g = TileGrid(tile=160, stride=120, image_size=640)
    assert list(g.origins_1d) == [0, 120, 240, 360, 480]
    assert g.grid_shape == (5, 5) and g.n_tiles == 25
    assert g.overlap == 40
    t = g.tiles()
    assert t.shape == (25, 4)
    # row-major: tile 1 is the second column of the first row
    assert list(t[0]) == [0, 0, 160, 160]
    assert list(t[1]) == [120, 0, 280, 160]
    assert list(t[5]) == [0, 120, 160, 280]
    # every tile is inside the image
    assert t[:, 2].max() == 640 and t[:, 3].max() == 640


def test_grid_last_tile_is_flush_when_stride_does_not_divide():
    g = TileGrid(tile=160, stride=100, image_size=640)
    # 0,100,...,400 then a short final step to 480 so the edge is covered
    assert list(g.origins_1d) == [0, 100, 200, 300, 400, 480]
    assert g.tiles()[:, 2].max() == 640


def test_grid_no_overlap_case():
    g = TileGrid(tile=128, stride=128, image_size=640)
    assert g.n_tiles == 25 and g.overlap == 0
    assert g.pixel_cost == 640 * 640


def test_cut_matches_tile_coordinates():
    g = TileGrid(tile=160, stride=120)
    img = np.arange(640 * 640, dtype=np.int32).reshape(640, 640)
    tiles = g.cut(img)
    assert tiles.shape == (25, 160, 160)
    for k, (x1, y1, x2, y2) in enumerate(g.tiles()):
        assert np.array_equal(tiles[k], img[y1:y2, x1:x2])


def test_overlap_frac_is_fraction_of_box_area():
    tiles = np.array([[0, 0, 100, 100]])
    # box fully inside
    assert box_overlap_fracs(np.array([[10, 10, 20, 20, 1]]), tiles)[0, 0] == 1.0
    # exactly half the box inside (x 90..110)
    assert box_overlap_fracs(np.array([[90, 10, 110, 20, 1]]), tiles)[0, 0] == 0.5
    # a quarter (both axes half)
    assert box_overlap_fracs(np.array([[90, 90, 110, 110, 1]]), tiles)[0, 0] == 0.25
    # disjoint, and touching only at the half-open edge
    assert box_overlap_fracs(np.array([[200, 200, 210, 210, 1]]), tiles)[0, 0] == 0.0
    assert box_overlap_fracs(np.array([[100, 0, 110, 10, 1]]), tiles)[0, 0] == 0.0


def test_label_positive_ignore_and_clean():
    g = TileGrid(tile=160, stride=120)
    # A box well inside tile 0 only.
    boxes = np.array([[20, 20, 60, 60, 3]])
    y, ign = label_tiles(boxes, g)
    ch = CLASS_ID_TO_CHANNEL[3]
    assert y[0, ch] == 1.0 and not ign[0, ch]
    assert y.sum() == 1.0            # no other tile claims it
    assert ign.sum() == 0            # it is not even partially in any other tile
    clean = is_clean(y, ign)
    assert clean[0] == False and clean.sum() == 24  # noqa: E712


def test_label_fragment_becomes_ignore_not_negative():
    g = TileGrid(tile=160, stride=120)
    # Box spans x 150..170: 10/20 = 50% in tile 0, 20/20 = 100% in tile 1.
    boxes = np.array([[150, 20, 170, 40, 2]])
    y, ign = label_tiles(boxes, g, pos_thresh=0.7)
    ch = CLASS_ID_TO_CHANNEL[2]
    assert y[1, ch] == 1.0 and not ign[1, ch]       # tile 1 fully contains it
    assert y[0, ch] == 0.0 and ign[0, ch]           # tile 0 only sees half
    # the ignore flag never coexists with a positive target
    assert not (ign & (y > 0)).any()


def test_ignore_threshold_turns_slivers_back_into_negatives():
    g = TileGrid(tile=160, stride=120)
    # 1 px of a 20 px box pokes into tile 0 (x 159..179 -> 1/20 = 5%).
    boxes = np.array([[159, 20, 179, 40, 4]])
    ch = CLASS_ID_TO_CHANNEL[4]
    _, ign_strict = label_tiles(boxes, g, ignore_thresh=0.0)
    assert ign_strict[0, ch]
    _, ign_loose = label_tiles(boxes, g, ignore_thresh=0.10)
    assert not ign_loose[0, ch]


def test_multilabel_independent_channels():
    g = TileGrid(tile=160, stride=120)
    boxes = np.array([[20, 20, 60, 60, 1], [70, 70, 110, 110, 5]])
    y, ign = label_tiles(boxes, g)
    assert y[0, CLASS_ID_TO_CHANNEL[1]] == 1.0
    assert y[0, CLASS_ID_TO_CHANNEL[5]] == 1.0
    assert y[0].sum() == 2.0
    assert y.shape[1] == NUM_CLASSES and not ign[0].any()


def test_empty_annotation_gives_all_clean():
    g = TileGrid(tile=160, stride=120)
    y, ign = label_tiles(np.zeros((0, 5), dtype=np.int32), g)
    assert y.shape == (25, 6) and not y.any() and not ign.any()
    assert is_clean(y, ign).all()


def test_gate_labels_follow_stage2_labels():
    g = TileGrid(tile=160, stride=120)
    boxes = np.array([[150, 20, 170, 40, 2]])
    y, ign = label_tiles(boxes, g)
    gy, gign = gate_labels(y, ign)
    assert gy[1] == 1.0 and not gign[1]      # positive tile -> gate must fire
    assert gy[0] == 0.0 and gign[0]          # fragment-only tile -> ignored
    assert gy.sum() == 1.0 and gign.sum() == 1


def test_best_tile_per_box():
    g = TileGrid(tile=160, stride=120)
    boxes = np.array([[150, 20, 170, 40, 2], [20, 20, 60, 60, 1]])
    idx, frac = best_tile_per_box(boxes, g)
    assert list(idx) == [1, 0]
    assert np.allclose(frac, [1.0, 1.0])


def test_invalid_configs_rejected():
    for kwargs in ({"tile": 0}, {"tile": 700}, {"stride": 0}, {"tile": 100, "stride": 200}):
        try:
            TileGrid(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"TileGrid({kwargs}) should have raised")
    try:
        label_tiles(np.zeros((0, 5)), TileGrid(), pos_thresh=0.5, ignore_thresh=0.6)
    except ValueError:
        pass
    else:
        raise AssertionError("ignore_thresh >= pos_thresh should have raised")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
