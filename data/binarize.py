"""Shared binarization: one algorithm for dataset images and camera frames.

Everything downstream -- the Stage-1 gate, the tile labels, Stage 2's input --
sees a binary image. If the training data and the camera are binarized
differently, the model is evaluated on one distribution and deployed on
another, so this module and ``firmware/pcb_bench/main/pcb_binarize.cpp`` must
agree bit for bit. ``tests/test_binarize_parity.py`` checks that they do.

The pipeline, in order:

1. **Channel.** Take one channel of the RGB image. For PKU-Market-PCB this is
   **green**, not red. Red separates the tinned pads from the board best (Otsu
   bimodality 0.90 vs 0.64), which is the obvious criterion and the wrong one:
   five of the six defect classes sit on traces *under* the green solder mask,
   which red cannot see. On red, defect survival after the gate's erosion is
   0-4.5% for every class except missing_hole; on green it is 87.5-100%.
   Copper under mask reflects more green than bare laminate, so the green
   channel recovers a trace map that looks like DeepPCB's.
2. **Flat field** (camera only). Subtract a box-blurred copy and re-centre:
   ``g - blur(g) + 128``. This removes the illumination gradient a desk lamp
   and the lens's own vignetting put across the frame, which otherwise moves
   the global threshold around the board. Dataset photos are evenly lit and
   have no flat-field frame, so they skip this step -- everything after it is
   identical for both paths. Subtractive, not multiplicative, because a divide
   per pixel is expensive on the chip and the correction only has to remove a
   low-frequency ramp.
3. **Global Otsu**, computed on 256 integer bins (see :func:`otsu_threshold`).
4. **Threshold and polarity.** ``g > t`` gives copper = 1 on these photos.
   DeepPCB's own files are the other way round -- its copper is *black*, and
   ``deeppcb.load_binary`` returns 1 for the substrate -- so the PKU converter
   passes ``invert=True`` to match the convention the existing models were
   trained on. The docstring in ``data/deeppcb.py`` claiming copper is white is
   wrong; the polarity is what matters here, so it is stated explicitly.

**No floating point anywhere.** Otsu is usually written with doubles, but the
host compiler is free to contract those into fused multiply-adds while the
Xtensa build does not, which would silently move the threshold by one level on
some frames. Every step below is integer arithmetic that both languages
evaluate identically.
"""

from __future__ import annotations

import math

import numpy as np

#: Channel index into an RGB image. Green: see the module docstring.
CHANNELS = {"r": 0, "g": 1, "b": 2}
DEFAULT_CHANNEL = "g"

#: Box-blur radius for the flat-field step, in pixels.
FLAT_FIELD_RADIUS = 32


def to_channel(image: np.ndarray, channel: str = DEFAULT_CHANNEL) -> np.ndarray:
    """One ``(H, W) uint8`` channel from an RGB or already-gray image."""
    if image.ndim == 2:
        return np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected (H, W) or (H, W, 3), got {image.shape}")
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {sorted(CHANNELS)}, got {channel!r}")
    return np.ascontiguousarray(image[:, :, CHANNELS[channel]], dtype=np.uint8)


def box_blur(gray: np.ndarray, radius: int = FLAT_FIELD_RADIUS) -> np.ndarray:
    """Mean over a ``(2r+1)^2`` window, edges clamped, integer truncation.

    Computed from an integral image so the cost does not grow with the radius,
    which is how the firmware does it. The division truncates towards zero in
    both languages (all values are non-negative), so the result is identical.
    """
    if radius <= 0:
        return np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape
    pad = np.pad(gray.astype(np.int64), radius, mode="edge")
    integral = np.zeros((h + 2 * radius + 1, w + 2 * radius + 1), dtype=np.int64)
    integral[1:, 1:] = pad.cumsum(axis=0).cumsum(axis=1)
    side = 2 * radius + 1
    total = (integral[side:, side:] - integral[:-side, side:]
             - integral[side:, :-side] + integral[:-side, :-side])
    return (total // (side * side)).astype(np.uint8)


def flat_field(gray: np.ndarray, radius: int = FLAT_FIELD_RADIUS) -> np.ndarray:
    """``gray - blur(gray) + 128``, clamped to 0..255.

    Within ``radius`` of an edge the clamped window cannot see the illumination
    field continue, so a gradient is only partly removed there. The inspected
    region is a central crop, so this does not reach it; it is left alone
    rather than fixed because any fix has to be mirrored on the chip.
    """
    out = gray.astype(np.int64) - box_blur(gray, radius).astype(np.int64) + 128
    return np.clip(out, 0, 255).astype(np.uint8)


def histogram(gray: np.ndarray) -> np.ndarray:
    """256 bins, ``int64``."""
    return np.bincount(gray.ravel(), minlength=256).astype(np.int64)


def otsu_threshold(gray_or_hist: np.ndarray) -> int:
    """Otsu's threshold, integer-only, in the exact form the firmware uses.

    Otsu maximizes the between-class variance ``w0 * w1 * (m0 - m1)^2``. Held
    that way it cannot be computed in int64 on a megapixel frame: ``w0 * w1``
    alone reaches 1.4e13, and multiplying by a usefully precise squared mean
    difference overflows. Every fixed-point arrangement tried here (Q8 means
    with shifts, or dividing the weights by the pixel count) threw away enough
    precision to move the threshold inside a *populated* part of the
    histogram, which changes real pixels.

    So it is rearranged instead. With

        A = s0 * w1 - (sum_total - s0) * w0        (= w0 * w1 * (m0 - m1))

    the criterion is ``A^2 / (w0 * w1)``, and ordering by that is the same as
    ordering by ``|A| / sqrt(w0 * w1)``, which is computed as

        q = (|A| << 8) // isqrt((w0 * w1) << 16)

    -- one integer division and one exact integer square root. Both shifts are
    needed: ``isqrt`` of the bare product truncates to about one part in
    ``sqrt(w0 * w1)``, which is 2e-4 on a small frame and swamps everything
    else, while scaling the numerator alone does nothing because the error is
    in the denominator. Scaled this way the result tracks an exact rational
    Otsu to ~1e-6.

    The shifts set the frame-size ceiling: the largest intermediate is
    ``(w0 * w1) << 16``, which reaches 9.3e17 on a 3056x2464 photo -- the
    largest in PKU-Market-PCB -- against an int64 limit of 9.2e18. Frames
    above about 8 megapixels would need the shifts reduced on both sides.

    What this does *not* do is always pick the same level as a float
    implementation. When two modes are well separated, every threshold in the
    empty valley between them scores within ~1e-5 of the best, so the argmax
    there is arbitrary; see tests/test_binarize.py. That is a property of
    Otsu's criterion, not of this arithmetic.

    Ties keep the lowest threshold: both implementations scan upwards with a
    strict ``>``. A frame with only one grey level has no valid split; both
    then return 255, so a featureless frame binarizes to all-background rather
    than all-copper.

    Returns the threshold ``t``; the foreground is ``gray > t``.
    """
    hist = (gray_or_hist.astype(np.int64) if gray_or_hist.ndim == 1
            else histogram(gray_or_hist))
    if hist.shape != (256,):
        raise ValueError(f"expected a 256-bin histogram, got {hist.shape}")

    total = int(hist.sum())
    if total == 0:
        return 255
    sum_total = int((hist * np.arange(256, dtype=np.int64)).sum())

    w0 = 0
    s0 = 0
    best_score = -1
    best_t = 255
    for t in range(256):
        w0 += int(hist[t])
        s0 += t * int(hist[t])
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        a = abs(s0 * w1 - (sum_total - s0) * w0)
        score = (a << 8) // math.isqrt((w0 * w1) << 16)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


def binarize(
    image: np.ndarray,
    channel: str = DEFAULT_CHANNEL,
    use_flat_field: bool = False,
    invert: bool = False,
    threshold: int | None = None,
    radius: int = FLAT_FIELD_RADIUS,
) -> tuple[np.ndarray, int]:
    """Binarize one image. Returns ``({0,1} uint8 array, threshold used)``.

    Args:
        image: ``(H, W, 3)`` RGB or ``(H, W)`` gray, ``uint8``.
        channel: which channel to threshold; green by default.
        use_flat_field: on for camera frames, off for dataset photos.
        invert: return ``1`` where the channel is *below* the threshold. The
            PKU converter sets this so copper is 0, matching DeepPCB's files.
        threshold: skip Otsu and use this level. The converter thresholds a
            test image with its *template's* level, so a defect cannot move the
            operating point; the firmware does the same against the stored
            reference.
        radius: flat-field box-blur radius.
    """
    gray = to_channel(image, channel)
    if use_flat_field:
        gray = flat_field(gray, radius)
    t = otsu_threshold(gray) if threshold is None else int(threshold)
    mask = (gray <= t) if invert else (gray > t)
    return mask.astype(np.uint8), t
