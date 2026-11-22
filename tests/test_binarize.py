"""The shared binarization, checked without a compiler.

tests/test_binarize_parity.py checks the firmware agrees; this file checks the
Python side is right in the first place.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "data"))

from binarize import (  # noqa: E402
    binarize,
    binarize_board,
    box_blur,
    flat_field,
    histogram,
    otsu_threshold,
    to_channel,
)


def brute_force_otsu(hist: np.ndarray) -> int:
    """Textbook float Otsu, for comparison on well-separated histograms."""
    total = hist.sum()
    sum_total = (hist * np.arange(256)).sum()
    best_t, best = 0, -1.0
    w0 = s0 = 0.0
    for t in range(256):
        w0 += hist[t]
        s0 += t * hist[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = s0 / w0, (sum_total - s0) / w1
        score = w0 * w1 * (m0 - m1) ** 2
        if score > best:
            best, best_t = score, t
    return best_t


def test_channel_selection():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30
    assert to_channel(img, "r")[0, 0] == 10
    assert to_channel(img, "g")[0, 0] == 20
    assert to_channel(img, "b")[0, 0] == 30
    assert to_channel(np.full((4, 4), 7, np.uint8))[0, 0] == 7
    with pytest.raises(ValueError):
        to_channel(img, "x")


def test_otsu_splits_two_clear_modes():
    """A histogram with two separated spikes must threshold between them."""
    hist = np.zeros(256, np.int64)
    hist[40] = 5000
    hist[200] = 3000
    t = otsu_threshold(hist)
    assert 40 <= t < 200


def float_score_curve(hist: np.ndarray) -> np.ndarray:
    """Otsu's between-class variance per threshold, in float, for comparison."""
    total = hist.sum()
    sum_total = (hist * np.arange(256)).sum()
    out = np.full(256, -1.0)
    w0 = s0 = 0.0
    for t in range(256):
        w0 += hist[t]
        s0 += t * hist[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        out[t] = w0 * w1 * (s0 / w0 - (sum_total - s0) / w1) ** 2
    return out


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("separation", ["wide", "overlapping"])
def test_integer_otsu_is_as_good_as_float_otsu(seed, separation):
    """The integer threshold must score as well as the float one.

    Deliberately not "must equal the float threshold". When two modes are well
    separated every level in the empty valley between them scores within ~1e-5
    of the best, so Otsu's own criterion does not distinguish them and the
    argmax there is arbitrary -- float and integer can land tens of levels
    apart and both are right. What must hold is that the chosen threshold is
    not measurably worse by the criterion being optimized.
    """
    rng = np.random.default_rng(seed)
    if separation == "wide":
        dark, bright = rng.normal(50, 12, (120, 160)), rng.normal(200, 15, (120, 160))
    else:
        dark, bright = rng.normal(95, 35, (120, 160)), rng.normal(135, 35, (120, 160))
    gray = np.where(rng.random((120, 160)) < 0.4, bright, dark)
    gray = gray.clip(0, 255).astype(np.uint8)

    curve = float_score_curve(histogram(gray))
    got = otsu_threshold(gray)
    best = float(curve.max())
    assert curve[got] >= best * (1 - 1e-5), (
        f"integer threshold {got} scores {curve[got]:.6e}, "
        f"float best {int(curve.argmax())} scores {best:.6e}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_integer_otsu_tracks_an_exact_rational_otsu(seed):
    """Against an Otsu with no rounding at all, the int64 form must not drift.

    This is the check the float comparison cannot give: the reference below is
    exact rational arithmetic, so any score gap is the int64 rearrangement
    losing precision and nothing else. The tolerance covers flat peaks, where
    the argmax is arbitrary; it caught two earlier arrangements that were out
    by 2e-4.
    """
    from fractions import Fraction

    rng = np.random.default_rng(seed)
    gray = np.where(rng.random((90, 110)) < 0.45,
                    rng.normal(70, 20, (90, 110)),
                    rng.normal(170, 25, (90, 110))).clip(0, 255).astype(np.uint8)
    hist = [int(v) for v in histogram(gray)]

    total = sum(hist)
    sum_total = sum(i * h for i, h in enumerate(hist))
    w0 = s0 = 0
    curve: dict[int, Fraction] = {}
    for t in range(256):
        w0 += hist[t]
        s0 += t * hist[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        curve[t] = Fraction((s0 * w1 - (sum_total - s0) * w0) ** 2, w0 * w1)

    best = max(curve.values())
    got = otsu_threshold(gray)
    loss = float((best - curve[got]) / best)
    assert loss < 1e-5, f"threshold {got} loses {loss:.3e} of the exact best score"


def test_otsu_on_a_flat_image_is_harmless():
    gray = np.full((16, 16), 99, np.uint8)
    t = otsu_threshold(gray)
    mask, used = binarize(gray, threshold=None)
    assert used == t
    assert mask.sum() == 0            # nothing is strictly above a flat level


def test_histogram_counts_every_pixel():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (37, 41)).astype(np.uint8)
    hist = histogram(gray)
    assert hist.sum() == gray.size
    assert hist[int(gray[0, 0])] >= 1


def test_box_blur_matches_a_naive_window():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, (23, 29)).astype(np.uint8)
    radius = 3
    got = box_blur(gray, radius)

    pad = np.pad(gray.astype(np.int64), radius, mode="edge")
    side = 2 * radius + 1
    want = np.zeros_like(gray)
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            want[y, x] = pad[y:y + side, x:x + side].sum() // (side * side)
    assert np.array_equal(got, want)


def test_box_blur_of_a_constant_is_that_constant():
    gray = np.full((20, 20), 123, np.uint8)
    assert np.array_equal(box_blur(gray, 4), gray)


def test_flat_field_removes_a_linear_ramp():
    """An illumination gradient must not survive; the texture must."""
    h, w = 160, 160
    ramp = np.linspace(40, 210, w, dtype=np.float64)[None, :].repeat(h, axis=0)
    texture = np.zeros((h, w))
    texture[:, ::8] = 45                      # thin bright lines
    gray = np.clip(ramp + texture, 0, 255).astype(np.uint8)

    radius = 16
    corrected = flat_field(gray, radius=radius)
    # Away from the borders the ramp must be gone. Inside `radius` of an edge
    # the clamped window cannot see the ramp continue, so a residual there is
    # expected and is documented in data/binarize.py.
    interior = corrected[:, radius:-radius].astype(np.float64)
    before = gray[:, radius:-radius].astype(np.float64)
    # Look at the background columns only: the bright lines every 8 px are
    # texture, and including them would measure the texture, not the ramp.
    background = np.array([x for x in range(interior.shape[1])
                           if (x + radius) % 8 != 0])
    means = interior[:, background].mean(axis=0)
    assert means.max() - means.min() < 20, "the ramp survived flat-fielding"
    assert np.ptp(before[:, background].mean(axis=0)) > 100, "the ramp was never there"
    # The lines must still be there.
    assert corrected[:, ::8].mean() > corrected[:, 1::8].mean() + 10


def test_polarity_and_explicit_threshold():
    gray = np.array([[10, 200], [10, 200]], np.uint8)
    mask, t = binarize(gray, threshold=100)
    assert t == 100
    assert np.array_equal(mask, np.array([[0, 1], [0, 1]], np.uint8))

    inverted, _ = binarize(gray, threshold=100, invert=True)
    assert np.array_equal(inverted, 1 - mask)


def test_binarize_uses_the_requested_channel():
    """Green and red must disagree when only one carries the structure."""
    img = np.zeros((8, 8, 3), np.uint8)
    img[..., 0] = 120                           # red: flat
    img[..., 1] = 60
    img[:, 4:, 1] = 200                         # green: a step
    mask_g, _ = binarize(img, channel="g")
    assert mask_g[:, :4].sum() == 0
    assert mask_g[:, 4:].all()


def test_binarize_board_needs_rgb():
    with pytest.raises(ValueError):
        binarize_board(np.zeros((8, 8), np.uint8))


def test_binarize_board_finds_both_kinds_of_copper():
    """Exposed pads (bright red) and masked traces (green over laminate)."""
    h = w = 64
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 8                       # green solder mask: red is nearly dark
    img[..., 1] = 60                      # bare laminate under mask
    img[20:30, 5:55, 1] = 140             # a trace: more green
    img[40:50, 40:50] = (200, 200, 200)   # a tinned pad: bright everywhere

    mask, (t_red, t_green) = binarize_board(img)
    copper = mask == 0
    assert copper[44, 44], "the exposed pad must be copper"
    assert copper[25, 30], "the masked trace must be copper"
    assert not copper[5, 5], "bare laminate must not be copper"
    assert 8 <= t_red < 200 and 60 <= t_green < 140


def test_binarize_board_ones_are_non_copper():
    """DeepPCB's convention: its copper is black, so ones mean substrate."""
    img = np.zeros((16, 16, 3), np.uint8)
    img[..., 0] = 5
    img[..., 1] = 40
    img[:, 8:, 1] = 200                   # right half is copper under mask
    mask, _ = binarize_board(img)
    assert mask[:, :8].all() and not mask[:, 8:].any()


def test_binarize_board_accepts_fixed_thresholds():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (24, 24, 3)).astype(np.uint8)
    mask, used = binarize_board(img, thresholds=(120, 90))
    assert used == (120, 90)
    expected = ~((img[..., 0] > 120) | ((img[..., 1] > 90) & (img[..., 0] <= 120)))
    assert np.array_equal(mask, expected.astype(np.uint8))
