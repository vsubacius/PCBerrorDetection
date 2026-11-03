#include "board.hpp"

#include "esp_heap_caps.h"

namespace pcb {
namespace {

// Small deterministic PRNG: we want the same synthetic board on every run and
// on every machine, which rand() does not promise.
struct Lcg {
    uint32_t state;
    uint32_t next()
    {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    uint32_t below(uint32_t bound) { return next() % bound; }
};

void draw_rect(uint8_t *image, int x, int y, int w, int h, uint8_t value)
{
    for (int yy = y; yy < y + h && yy < kBoardSize; ++yy) {
        if (yy < 0) {
            continue;
        }
        for (int xx = x; xx < x + w && xx < kBoardSize; ++xx) {
            if (xx >= 0) {
                image[static_cast<size_t>(yy) * kBoardSize + xx] = value;
            }
        }
    }
}

}  // namespace

bool alloc_board(Board *board)
{
    const size_t bytes = static_cast<size_t>(kBoardSize) * kBoardSize;
    board->test = static_cast<uint8_t *>(heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM));
    board->temp = static_cast<uint8_t *>(heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM));
    if (board->test == nullptr || board->temp == nullptr) {
        free_board(board);
        return false;
    }
    return true;
}

void free_board(Board *board)
{
    heap_caps_free(board->test);
    heap_caps_free(board->temp);
    board->test = nullptr;
    board->temp = nullptr;
}

void make_synthetic_board(const Board &board, uint32_t seed, int n_defects)
{
    const size_t bytes = static_cast<size_t>(kBoardSize) * kBoardSize;
    Lcg rng{seed};

    // Copper background, then etched channels, to land near DeepPCB's ~81%
    // copper coverage with a realistic amount of trace edge (the edge fringe is
    // what the gate's erosion has to survive).
    for (size_t i = 0; i < bytes; ++i) {
        board.temp[i] = 1;
    }
    for (int i = 0; i < 160; ++i) {
        const int x = static_cast<int>(rng.below(kBoardSize));
        const int y = static_cast<int>(rng.below(kBoardSize));
        if (rng.next() & 1u) {
            draw_rect(board.temp, x, y, 6, 60 + static_cast<int>(rng.below(120)), 0);
        } else {
            draw_rect(board.temp, x, y, 60 + static_cast<int>(rng.below(120)), 6, 0);
        }
    }

    // The test image is the template plus defects, which is what makes most
    // tiles clean and therefore gateable.
    for (size_t i = 0; i < bytes; ++i) {
        board.test[i] = board.temp[i];
    }
    for (int i = 0; i < n_defects; ++i) {
        const int x = static_cast<int>(rng.below(kBoardSize - 40));
        const int y = static_cast<int>(rng.below(kBoardSize - 40));
        const int side = 24 + static_cast<int>(rng.below(16));
        draw_rect(board.test, x, y, side, side, (rng.next() & 1u) ? 1 : 0);
    }
}

}  // namespace pcb
