// Shared binarization -- the chip's half. Mirrors data/binarize.py exactly.
//
// Why this file exists. The model is trained on binarized dataset photos and
// deployed on binarized camera frames. If the two binarizations differ, the
// network sees a different distribution at inference than it was trained on,
// and no amount of gate tuning fixes that. So one algorithm is written twice,
// and tests/test_binarize_parity.py compiles this file on the host and checks
// it agrees with the Python reference pixel for pixel on real PKU crops.
//
// The pipeline: pick a channel (green -- see data/binarize.py for the measured
// reason), optionally flat-field the frame, take a global Otsu threshold, then
// threshold with the requested polarity.
//
// No floating point. Otsu is normally written with doubles, but the host
// compiler may contract those into fused multiply-adds where the Xtensa build
// will not, which moves the threshold by a level on some frames and breaks
// parity in a way that is very hard to see. Everything here is int64 integer
// arithmetic with fixed shifts, which both toolchains evaluate identically.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace pcb {

// Flat-field box-blur radius, in pixels. Must match FLAT_FIELD_RADIUS in
// data/binarize.py.
constexpr int kFlatFieldRadius = 32;

// Scratch bytes binarize() needs: a blurred copy of the frame plus one int32
// column sum per pixel of width. Zero when flat-fielding is off.
constexpr size_t binarize_scratch_bytes(int width, int height, bool flat_field)
{
    return flat_field ? (static_cast<size_t>(width) * static_cast<size_t>(height)
                         + static_cast<size_t>(width) * sizeof(int32_t))
                      : 0u;
}

// 256-bin histogram of an 8-bit image.
void histogram(const uint8_t *gray, int width, int height, int stride,
               uint32_t hist[256]);

// Otsu's threshold from a 256-bin histogram. The foreground is `gray > t`.
//
// Otsu maximizes w0 * w1 * (mean0 - mean1)^2, which does not fit in int64 at
// useful precision on a megapixel frame. With
//
//     A = s0 * w1 - (sum_total - s0) * w0        (= w0 * w1 * (m0 - m1))
//
// the criterion is A^2 / (w0 * w1), and ordering by that is ordering by
//
//     score = (|A| << 8) / isqrt((w0 * w1) << 16)
//
// -- one integer division and one exact integer square root. Both shifts
// matter; see data/binarize.py for the measured reason and for the ~8
// megapixel frame-size ceiling they imply.
//
// Scans upwards with a strict >, so ties keep the lowest threshold. A frame
// with a single grey level has no valid split and returns 255, so a
// featureless frame binarizes to all-background rather than all-copper.
int otsu_threshold(const uint32_t hist[256]);

// Mean over a (2r+1)^2 window with the edges clamped and the division
// truncating -- numpy's pad(mode="edge") followed by a box mean.
//
// Streams a ring of column sums instead of building an integral image, so the
// extra memory is one int32 per column rather than one per pixel. The sums are
// exact integers either way, so the result is identical to the Python side.
void box_blur(const uint8_t *src, int width, int height, int stride, int radius,
              uint8_t *dst, int32_t *col_sums);

// gray - blur(gray) + 128, clamped to 0..255, written in place into dst.
void flat_field(const uint8_t *src, int width, int height, int stride,
                int radius, uint8_t *dst, int32_t *col_sums);

// Binarize one frame to {0, 1} bytes.
//
// `gray` is a single channel already extracted from the camera frame (green
// for the PKU-trained model). `threshold` < 0 runs Otsu on the frame;
// otherwise the given level is used, which is what the live path does -- it
// thresholds against the level stored with the golden template so a defect
// cannot move the operating point. `invert` makes copper 0, matching DeepPCB's
// files and therefore the trained model's expectation.
//
// `scratch` must hold binarize_scratch_bytes(width, height, use_flat_field).
// Returns the threshold actually used.
int binarize(const uint8_t *gray, int width, int height, int stride,
             bool use_flat_field, bool invert, int threshold,
             uint8_t *dst, uint8_t *scratch);

// Copper map of a solder-masked board, from separate red and green planes.
//
// dst is 1 where there is NO copper, matching DeepPCB's files (its copper is
// black) and therefore what the trained model expects.
//
//     copper = (red > t_red) | ((green > t_green) & (red <= t_red))
//
// Red finds exposed metal -- tinned pads read ~126 in red against ~5 for
// green solder mask. Green then finds copper under the mask, which reflects
// more green than bare laminate. A single global Otsu on green is cheaper and
// was tried first; it fails outright on one of the ten PKU boards, where the
// threshold lands inside the copper distribution and defects on traces
// produce no XOR at all. data/binarize.py carries the measurements.
//
// t_red < 0 or t_green < 0 runs Otsu for that threshold -- the green one over
// the masked pixels only. The live path passes both levels from the stored
// template instead, so a defect cannot move the operating point.
//
// Needs binarize_board_scratch_bytes() of scratch. Returns nothing; the
// thresholds used are written back through t_red_out / t_green_out when those
// are non-null.
constexpr size_t binarize_board_scratch_bytes(int width, int height, bool flat_field)
{
    // Two planes to flat-field instead of one, plus the column sums.
    return flat_field ? (2u * static_cast<size_t>(width) * static_cast<size_t>(height)
                         + static_cast<size_t>(width) * sizeof(int32_t))
                      : 0u;
}

void binarize_board(const uint8_t *red, const uint8_t *green, int width, int height,
                    int stride, bool use_flat_field, int t_red, int t_green,
                    uint8_t *dst, uint8_t *scratch,
                    int *t_red_out = nullptr, int *t_green_out = nullptr);

}  // namespace pcb
