// Stage-1 reference-guided gate: count the eroded XOR of a tile against its
// template, and skip the tile if the count is at or below a threshold.
//
// Why erode first. DeepPCB's images are binary but JPEG-compressed, so
// re-thresholding leaves a 1-px fringe of disagreement along every copper
// edge. A median *clean* tile already carries ~165 raw XOR pixels, which
// swamps small defects: raw XOR counting separates clean from defective tiles
// at only AUC 0.78. A 3x3 erosion removes the fringe (a 1-px halo has no
// thick core) and takes AUC to 0.95, lifting the share of clean tiles skipped
// at 98% defect-tile recall from 44% to 88%.
//
// Do not raise the radius to 2. A 5x5 erosion also deletes 15.4% of *defect*
// tiles outright -- they score exactly zero, so no threshold can recover them
// and tile recall is capped near 85%. radius 1 leaves 0.0% of defect tiles at
// zero. (Measured on 60 validation images; see eval/gate_analysis.py.)
//
// Cost. A kxk binary erosion is a sliding AND, separable into a horizontal and
// a vertical pass, with no multiplies. This implementation streams the
// vertical pass over a ring of (2*radius+1) rows, so it needs
// (2*radius+1) * tile bytes of scratch -- 480 bytes for a 160-px tile at
// radius 1 -- instead of buffering the whole XOR image (25 KB).
//
// Parity. Edge pixels are handled by clamping the sampling index, which is
// exactly numpy's `pad(mode="edge")` followed by a windowed minimum. That
// makes this bit-identical to `data/tile_dataset.binary_morph(dilate=False)`
// applied per tile, which is what tests/test_gate_parity.py checks.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace pcb {

// Side of one tile in pixels, and the gate's erosion radius (1 => 3x3).
constexpr int kTileSize = 160;
constexpr int kGateRadius = 1;

// Scratch bytes gate_score() needs for a given tile side and radius.
constexpr size_t gate_scratch_bytes(int tile = kTileSize, int radius = kGateRadius)
{
    return static_cast<size_t>(2 * radius + 1) * static_cast<size_t>(tile);
}

// Number of surviving pixels in erode(XOR(test, temp)) over one tile.
//
// `test` and `temp` point at the tile's top-left pixel inside a larger board
// image whose row pitch is `stride` bytes. Pixels must be 0 or 1. `scratch`
// must hold at least gate_scratch_bytes(tile, radius) bytes.
int gate_score(const uint8_t *test, const uint8_t *temp, int stride,
               int tile, int radius, uint8_t *scratch);

// The gate fires (run Stage 2) when the score exceeds the threshold, matching
// the `score > thr` convention used when thresholds are selected on validation.
inline bool gate_fires(int score, int threshold) { return score > threshold; }

}  // namespace pcb
