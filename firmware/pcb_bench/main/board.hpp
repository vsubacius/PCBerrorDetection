// A 640x640 binary board pair (defective test image + defect-free template)
// held in PSRAM, plus the 160/120 tile grid that the cascade walks.
//
// Benchmark mode will load real DeepPCB pairs from microSD. Until the SD path
// exists, make_synthetic_board() fabricates a board with roughly the right
// statistics (~81% copper, a handful of defect blobs) so that latency and
// memory numbers can be taken before the card reader is wired up. Synthetic
// boards are for *timing only* -- never for accuracy numbers.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace pcb {

constexpr int kBoardSize = 640;
constexpr int kTileStride = 120;
constexpr int kGridSide = 5;                         // (640 - 160) / 120 + 1
constexpr int kTilesPerBoard = kGridSide * kGridSide;  // 25

// Byte offset of tile `index` (row-major) inside a board image.
inline size_t tile_offset(int index)
{
    const int row = index / kGridSide;
    const int col = index % kGridSide;
    return static_cast<size_t>(row) * kTileStride * kBoardSize
           + static_cast<size_t>(col) * kTileStride;
}

struct Board {
    uint8_t *test;   // (640, 640) of 0/1, defective
    uint8_t *temp;   // (640, 640) of 0/1, defect-free template
};

// Allocate a board pair in PSRAM. Returns false if either allocation fails.
bool alloc_board(Board *board);
void free_board(Board *board);

// Fill a board with synthetic traces plus `n_defects` blobs in the test image.
// Deterministic for a given seed.
void make_synthetic_board(const Board &board, uint32_t seed, int n_defects);

}  // namespace pcb
