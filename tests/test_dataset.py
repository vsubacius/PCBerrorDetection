"""Tests for the tile datasets, especially augmentation determinism.

The property that matters most here is that a given item augments the same way
regardless of *when* or *in which worker* it is accessed. A single shared
Generator breaks that twice over: DataLoader workers are forked copies, so
every worker replays the same sequence, and an item's augmentation depends on
how many items were drawn before it.

These tests build a tiny synthetic fold rather than loading DeepPCB, so they
run without the dataset; one integration test uses the real fold if present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))

from tile_dataset import (  # noqa: E402
    AugmentConfig,
    TileData,
    TileImageDataset,
    TorchTileDataset,
    _rng_for,
    collate_tiles,
)
from tiling import TileGrid, label_tiles  # noqa: E402

GRID = TileGrid(160, 120)


def make_fold(n_images: int = 6, seed: int = 0) -> TileData:
    """A small synthetic fold with real geometry and a defect per image."""
    rng = np.random.default_rng(seed)
    test = rng.integers(0, 2, (n_images, 640, 640)).astype(np.uint8)
    temp = test.copy()
    boxes = []
    for i in range(n_images):
        # One mousebite, placed so it lands cleanly inside a tile.
        x0, y0 = 20 + 40 * i, 20
        temp[i, y0:y0 + 30, x0:x0 + 30] = 0      # the template lacks the defect
        boxes.append(np.array([[x0, y0, x0 + 30, y0 + 30, 3]], dtype=np.int32))
    ys, igs = zip(*(label_tiles(b, GRID) for b in boxes))
    return TileData(
        test=test, temp=temp, y=np.stack(ys), ignore=np.stack(igs),
        boxes=list(boxes),
        names=np.array([f"img{i}" for i in range(n_images)]),
        groups=np.array(["g0"] * n_images), grid=GRID,
        meta={"pos_thresh": 0.7, "ignore_thresh": 0.0},
    )


# --------------------------------------------------------------------------- #
# shapes and collation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("representation,channels", [("A", 1), ("B", 2), ("C", 2)])
def test_item_shapes(representation, channels):
    ds = TileImageDataset(make_fold(), representation)
    x, y, ignore = ds[0]
    assert x.shape == (25, channels, 160, 160)
    assert y.shape == (25, 6) and ignore.shape == (25, 6)
    assert x.dtype == torch.float32 and y.dtype == torch.float32


def test_collate_flattens_images_into_tiles():
    ds = TileImageDataset(make_fold(), "B")
    x, y, ignore = collate_tiles([ds[0], ds[1], ds[2]])
    assert x.shape == (75, 2, 160, 160)
    assert y.shape == (75, 6) and ignore.shape == (75, 6)


def test_works_with_a_dataloader():
    ds = TileImageDataset(make_fold(), "A", AugmentConfig.train_default())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, collate_fn=collate_tiles)
    x, y, ignore = next(iter(loader))
    assert x.shape == (50, 1, 160, 160)
    assert y.shape == (50, 6)


def test_len_is_images_not_tiles():
    fold = make_fold(n_images=4)
    assert len(TileImageDataset(fold)) == 4
    assert len(TorchTileDataset(fold)) == 4 * 25


# --------------------------------------------------------------------------- #
# augmentation determinism -- the worker-safety property
# --------------------------------------------------------------------------- #

def test_same_item_is_identical_however_it_is_reached():
    """Access order must not change what an item looks like.

    This is the regression test for the shared-Generator bug: with shared
    mutable RNG state, reading item 3 after items 0-2 differed from reading it
    first.
    """
    fold = make_fold()
    aug = AugmentConfig.train_default()

    direct = TileImageDataset(fold, "B", aug, seed=0)[3]
    other = TileImageDataset(fold, "B", aug, seed=0)
    for k in (0, 1, 2, 4, 5):
        other[k]
    after = other[3]

    for a, b in zip(direct, after):
        assert torch.equal(a, b)


def test_two_instances_with_the_same_seed_agree_item_by_item():
    fold = make_fold()
    aug = AugmentConfig.train_default()
    a = TileImageDataset(fold, "C", aug, seed=7)
    b = TileImageDataset(fold, "C", aug, seed=7)
    for i in range(len(fold)):
        for xa, xb in zip(a[i], b[i]):
            assert torch.equal(xa, xb)


def test_multiple_dataloader_workers_do_not_duplicate_augmentation():
    """Each item must be augmented once, by its own generator.

    With a shared Generator every worker replayed the same stream, so the
    epoch contained duplicated transforms instead of independent ones.
    """
    fold = make_fold(n_images=8)
    ds = TileImageDataset(fold, "A", AugmentConfig.train_default(), seed=0)
    single = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0,
                                         collate_fn=collate_tiles)
    multi = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4,
                                        collate_fn=collate_tiles)
    got_single = [x.clone() for x, _, _ in single]
    got_multi = [x.clone() for x, _, _ in multi]
    assert len(got_single) == len(got_multi) == 8
    for a, b in zip(got_single, got_multi):
        assert torch.equal(a, b), "worker count must not change the data"


def test_set_epoch_rerolls_augmentation():
    fold = make_fold()
    ds = TileImageDataset(fold, "A", AugmentConfig.train_default(), seed=0)
    first = ds[0][0].clone()
    ds.set_epoch(1)
    second = ds[0][0].clone()
    assert not torch.equal(first, second), "a new epoch must re-roll transforms"
    ds.set_epoch(0)
    assert torch.equal(ds[0][0], first), "epochs must be reproducible"


def test_different_items_get_independent_transforms():
    fold = make_fold(n_images=12)
    ds = TileImageDataset(fold, "A", AugmentConfig.train_default(), seed=0)
    sums = {float(ds[i][0].sum()) for i in range(len(fold))}
    assert len(sums) > 1, "all items transformed identically"


def test_rng_depends_on_all_three_of_seed_epoch_and_index():
    base = _rng_for(0, 0, 0).random(4)
    assert not np.array_equal(base, _rng_for(1, 0, 0).random(4))
    assert not np.array_equal(base, _rng_for(0, 1, 0).random(4))
    assert not np.array_equal(base, _rng_for(0, 0, 1).random(4))
    assert np.array_equal(base, _rng_for(0, 0, 0).random(4))


def test_no_augmentation_is_deterministic_and_matches_the_raw_tiles():
    fold = make_fold()
    ds = TileImageDataset(fold, "B", aug=None, seed=0)
    x, y, ignore = ds[2]
    raw_x, raw_y, raw_ignore = fold.tiles(2, "B", None)
    assert np.array_equal(x.numpy(), raw_x)
    assert np.array_equal(y.numpy(), raw_y)
    assert np.array_equal(ignore.numpy(), raw_ignore.astype(np.float32))
    ds.set_epoch(5)
    assert torch.equal(ds[2][0], x), "without aug, the epoch must not matter"


# --------------------------------------------------------------------------- #
# labels survive augmentation
# --------------------------------------------------------------------------- #

def test_labels_follow_the_augmented_image():
    """Every image here has exactly one mousebite, whatever the transform."""
    fold = make_fold(n_images=10)
    ds = TileImageDataset(fold, "B", AugmentConfig.train_default(), seed=0)
    for i in range(len(fold)):
        _, y, ignore = ds[i]
        assert y[:, 2].sum() >= 1, "the mousebite label must survive"
        assert y.sum() == y[:, 2].sum(), "no other class may appear"
        assert not ((y > 0) & (ignore > 0)).any(), "ignore must not coexist with positive"


def test_ignore_mask_never_coexists_with_a_positive_label():
    fold = make_fold()
    ds = TileImageDataset(fold, "C", AugmentConfig.train_default(), seed=3)
    for i in range(len(fold)):
        _, y, ignore = ds[i]
        assert not ((y > 0) & (ignore > 0)).any()


# --------------------------------------------------------------------------- #
# tile subsampling
# --------------------------------------------------------------------------- #

def test_tiles_per_image_controls_the_item_size():
    ds = TileImageDataset(make_fold(), "A", tiles_per_image=8)
    x, y, ignore = ds[0]
    assert x.shape[0] == 8 and y.shape[0] == 8 and ignore.shape[0] == 8


def test_drop_clean_frac_removes_clean_tiles_only():
    fold = make_fold()
    ds = TileImageDataset(fold, "A", seed=0, drop_clean_frac=0.9)
    total_positive = 0
    for i in range(len(fold)):
        _, y, ignore = ds[i]
        assert y.shape[0] <= 25
        total_positive += int((y > 0).any(dim=1).sum())
    assert total_positive >= len(fold), "positive tiles must never be dropped"


def test_drop_clean_frac_never_yields_an_empty_item():
    fold = make_fold()
    ds = TileImageDataset(fold, "A", seed=0, drop_clean_frac=0.999)
    for i in range(len(fold)):
        x, _, _ = ds[i]
        assert x.shape[0] >= 1


@pytest.mark.parametrize("bad", [0, 26, -1])
def test_rejects_an_impossible_tiles_per_image(bad):
    with pytest.raises(ValueError):
        TileImageDataset(make_fold(), "A", tiles_per_image=bad)


@pytest.mark.parametrize("bad", [1.0, -0.1, 1.5])
def test_rejects_an_impossible_drop_clean_frac(bad):
    with pytest.raises(ValueError):
        TileImageDataset(make_fold(), "A", drop_clean_frac=bad)


# --------------------------------------------------------------------------- #
# the two datasets agree, and the fast one really is faster
# --------------------------------------------------------------------------- #

def test_per_tile_dataset_agrees_with_the_per_image_one_without_aug():
    fold = make_fold()
    per_image = TileImageDataset(fold, "B", aug=None)
    per_tile = TorchTileDataset(fold, "B", aug=None)
    x_img, y_img, ig_img = per_image[1]
    for t in range(fold.n_tiles):
        x_t, y_t, ig_t = per_tile[1 * fold.n_tiles + t]
        assert torch.equal(x_t, x_img[t])
        assert torch.equal(y_t, y_img[t])
        assert torch.equal(ig_t, ig_img[t])


def test_per_tile_access_is_also_worker_safe():
    fold = make_fold()
    aug = AugmentConfig.train_default()
    direct = TorchTileDataset(fold, "A", aug, seed=0)[40]
    other = TorchTileDataset(fold, "A", aug, seed=0)
    for k in range(10):
        other[k]
    for a, b in zip(direct, other[40]):
        assert torch.equal(a, b)


def test_class_pos_weights_are_finite_and_positive():
    weights = make_fold().class_pos_weights()
    assert weights.shape == (6,)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)


def test_real_fold_loads_and_serves_tiles():
    npz = _REPO / "data" / "tiles" / "deeppcb_tiles_t160_s120.npz"
    if not npz.exists():
        pytest.skip("tile dataset not built (python data/build_tiles.py)")
    fold = TileData.load(npz, "val")
    assert len(fold) == 200
    ds = TileImageDataset(fold, "A", AugmentConfig.train_default(), seed=0)
    x, y, ignore = ds[0]
    assert x.shape == (25, 1, 160, 160)
    assert x.min() >= 0.0 and x.max() <= 1.0
