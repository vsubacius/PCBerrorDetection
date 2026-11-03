#include "gate.hpp"

namespace pcb {
namespace {

inline int clamp_index(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// One row of the horizontal pass: AND of XOR(test, temp) over a window of
// 2*radius+1 columns, with the sampling index clamped at the tile edges.
void erode_row_horizontal(uint8_t *dst, const uint8_t *test_row,
                          const uint8_t *temp_row, int tile, int radius)
{
    for (int x = 0; x < tile; ++x) {
        uint8_t keep = 1;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xc = clamp_index(x + dx, 0, tile - 1);
            keep &= static_cast<uint8_t>(test_row[xc] ^ temp_row[xc]);
            if (!keep) {
                break;
            }
        }
        dst[x] = keep;
    }
}

}  // namespace

int gate_score(const uint8_t *test, const uint8_t *temp, int stride,
               int tile, int radius, uint8_t *scratch)
{
    if (tile <= 0 || radius < 0 || scratch == nullptr) {
        return 0;
    }
    const int window = 2 * radius + 1;

    // scratch is a ring of `window` horizontally-eroded rows; row j lives in
    // slot j % window. Rows are loaded in increasing order, and the vertical
    // pass for output row y reads clamped rows y-radius .. y+radius, which are
    // at most `window` consecutive indices and therefore occupy distinct slots.
    auto load_row = [&](int row) {
        erode_row_horizontal(scratch + static_cast<size_t>(row % window) * tile,
                             test + static_cast<size_t>(row) * stride,
                             temp + static_cast<size_t>(row) * stride,
                             tile, radius);
    };

    for (int row = 0; row <= clamp_index(radius, 0, tile - 1); ++row) {
        load_row(row);
    }

    int count = 0;
    for (int y = 0; y < tile; ++y) {
        const int next = y + radius;
        if (next < tile) {
            load_row(next);
        }
        for (int x = 0; x < tile; ++x) {
            uint8_t keep = 1;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int yc = clamp_index(y + dy, 0, tile - 1);
                keep &= scratch[static_cast<size_t>(yc % window) * tile + x];
                if (!keep) {
                    break;
                }
            }
            count += keep;
        }
    }
    return count;
}

}  // namespace pcb
