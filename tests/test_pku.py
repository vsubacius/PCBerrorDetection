"""The PKU converter: geometry, labels, and the DeepPCB interface it mimics.

The conversion is the one place where a silent error is invisible downstream --
a box scaled wrong or a template paired with the wrong board still trains, it
just trains on nonsense. These tests pin the parts that can be checked without
the dataset; the parts that cannot are checked by eye in
eval/pku_sanity_sheet.py and by the numbers in PKU_STEP1_REPORT.md.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))

import deeppcb  # noqa: E402
import pku  # noqa: E402

PKU_CROPS = _REPO / "data" / "PKU-crops"


def test_class_block_matches_deeppcb_interface():
    """build_tiles.py and grid_sweep.py use these interchangeably."""
    for attr in ("CLASS_IDS", "CLASS_NAMES", "NUM_CLASSES", "CLASS_ID_TO_CHANNEL",
                 "IMAGE_SIZE", "index", "Sample", "download"):
        assert hasattr(pku, attr), f"pku.{attr} missing"
    assert pku.CLASS_IDS == deeppcb.CLASS_IDS
    assert pku.NUM_CLASSES == deeppcb.NUM_CLASSES
    assert pku.IMAGE_SIZE == deeppcb.IMAGE_SIZE
    assert set(pku.VOC_CLASS_IDS.values()) == set(pku.CLASS_IDS)


def test_slot_six_is_missing_hole_not_pinhole():
    """The one class that does not transfer; it must not be silently merged."""
    assert pku.VOC_CLASS_IDS["missing_hole"] == 6
    assert pku.CLASS_NAMES[6] == "missing_hole"
    assert deeppcb.CLASS_NAMES[6] == "pin-hole"
    for name in ("open_circuit", "short", "mouse_bite", "spur", "spurious_copper"):
        cid = pku.VOC_CLASS_IDS[name]
        assert pku.CLASS_NAMES[cid] == deeppcb.CLASS_NAMES[cid]


def test_scale_table_is_consistent():
    assert pku.target_px_per_mm() == pytest.approx(57 / 2.54, abs=1e-6)
    for board, pitch in pku.BOARD_PITCH_PX.items():
        assert pku.board_scale(board) == pytest.approx(57 / pitch)
        assert pku.board_px_per_mm(board) == pytest.approx(pitch / 2.54)
    # The four 57 px boards are already at the target and must not be resampled.
    assert pku.board_scale("01") == 1.0


def test_crop_origins_are_flush_with_the_edge():
    assert pku.crop_origins(640) == [0]
    assert pku.crop_origins(639) == []                  # too small for one crop
    assert pku.crop_origins(1280) == [0, 640]
    origins = pku.crop_origins(1500)
    assert origins[0] == 0 and origins[-1] == 1500 - 640
    assert all(o + 640 <= 1500 for o in origins)


def test_boxes_in_crop_applies_the_seventy_percent_rule():
    size = pku.IMAGE_SIZE
    # Fully inside.
    box = np.array([[700, 700, 740, 740, 3]], np.int32)
    kept = pku.boxes_in_crop(box, 640, 640)
    assert len(kept) == 1
    assert kept[0].tolist() == [60, 60, 100, 100, 3]

    # Half outside -> dropped at keep_frac 0.7.
    straddle = np.array([[620, 700, 660, 740, 3]], np.int32)
    assert len(pku.boxes_in_crop(straddle, 640, 640)) == 0

    # 75% inside -> kept, and clipped to the crop.
    mostly = np.array([[630, 700, 670, 740, 3]], np.int32)
    kept = pku.boxes_in_crop(mostly, 640, 640)
    assert len(kept) == 1
    assert kept[0][0] == 0 and kept[0][2] == 30
    assert kept[0][2] <= size and kept[0][3] <= size

    assert len(pku.boxes_in_crop(np.zeros((0, 5), np.int32), 0, 0)) == 0


def test_boxes_in_crop_matches_the_tiling_rule():
    """Crop selection and tile labelling must use the same threshold."""
    from tiling import POS_THRESH
    assert pku.KEEP_FRAC == POS_THRESH


def test_best_shift_recovers_a_known_offset():
    rng = np.random.default_rng(0)
    temp = (rng.random((400, 400)) < 0.3).astype(np.uint8)
    for dx, dy in ((0, 0), (2, -3), (-1, 1)):
        test = np.roll(np.roll(temp, dy, axis=0), dx, axis=1)
        assert pku.best_shift(test, temp, radius=4, window=256) == (dx, dy)


def test_shift_image_round_trips():
    rng = np.random.default_rng(1)
    img = (rng.random((50, 60)) < 0.5).astype(np.uint8)
    assert np.array_equal(pku.shift_image(img, 0, 0), img)
    moved = pku.shift_image(img, 3, -2)
    assert np.array_equal(pku.shift_image(moved, -3, 2), img)


def test_measure_pitch_prefers_the_fundamental_over_its_harmonic():
    """Boards 01 and 05 peak at 2x and 3x their true pitch."""
    grid = np.zeros((600, 600), np.uint8)
    grid[::40, ::40] = 1                      # pads on a 40 px grid
    for y in range(0, 600, 40):
        for x in range(0, 600, 40):
            grid[y:y + 6, x:x + 6] = 1
    assert pku.measure_pitch(grid, lo=30, hi=200) == 40


def test_resample_scales_and_is_a_noop_at_one():
    arr = np.zeros((100, 200, 3), np.uint8)
    assert pku.resample(arr, 1.0) is arr
    assert pku.resample(arr, 0.5).shape == (50, 100, 3)


def test_read_annotations_round_trips(tmp_path):
    boxes = np.array([[10, 20, 30, 40, 1], [100, 100, 200, 220, 6]], np.int32)
    test = np.zeros((640, 640), np.uint8)
    temp = np.zeros((640, 640), np.uint8)
    test[5:10, 5:10] = 1
    pku.write_pair(tmp_path, "x", test, temp, boxes)
    assert np.array_equal(pku.read_annotations(tmp_path / "x.txt"), boxes)
    assert np.array_equal(pku.load_png(tmp_path / "x_test.png"), test)
    assert pku.load_png(tmp_path / "x_temp.png").sum() == 0


def test_read_annotations_rejects_out_of_range(tmp_path):
    (tmp_path / "bad.txt").write_text("0 0 700 10 1\n")
    with pytest.raises(ValueError):
        pku.read_annotations(tmp_path / "bad.txt")
    (tmp_path / "bad2.txt").write_text("0 0 10 10 9\n")
    with pytest.raises(ValueError):
        pku.read_annotations(tmp_path / "bad2.txt")


def test_empty_annotation_file_is_a_clean_crop(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    boxes = pku.read_annotations(tmp_path / "empty.txt")
    assert boxes.shape == (0, 5)


@pytest.mark.skipif(not (PKU_CROPS / "index.json").is_file(),
                    reason="PKU not converted (python data/pku.py --convert)")
def test_converted_dataset_is_consistent():
    samples = pku.index(PKU_CROPS)
    assert samples, "no converted samples"

    folds = {f: [s for s in samples if s.split == f] for f in ("train", "val", "test")}
    assert all(folds.values()), f"empty fold: { {k: len(v) for k, v in folds.items()} }"

    # No board may appear in two folds -- that is the leak the split exists to
    # prevent, since 60-odd photographs of one board share a layout.
    boards = {f: {s.board for s in v} for f, v in folds.items()}
    assert not (boards["train"] & boards["val"])
    assert not (boards["train"] & boards["test"])
    assert not (boards["val"] & boards["test"])

    meta = pku.meta(PKU_CROPS)
    assert set(meta["test_boards"]) == boards["test"]
    assert meta["ones_are"].startswith("non-copper")

    s = samples[0]
    test, temp = s.load()
    assert test.shape == temp.shape == (640, 640)
    assert set(np.unique(test)) <= {0, 1}
    for box in s.boxes():
        assert 0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 640
