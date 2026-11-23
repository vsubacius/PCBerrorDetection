#include "boards.hpp"

#include <string.h>

namespace pcb {

bool parse_board_blob(const uint8_t *blob, size_t size, BoardSet *out)
{
    *out = BoardSet{};
    if (blob == nullptr || size < sizeof(BlobHeader)) {
        return false;
    }
    const auto *header = reinterpret_cast<const BlobHeader *>(blob);
    if (memcmp(header->magic, "PCB1", 4) != 0 || header->version != 1) {
        return false;
    }
    const size_t n = header->n_boards;
    const size_t need = sizeof(BlobHeader) + n * sizeof(BlobBoardMeta)
                        + n * 2 * kPackedImageBytes;
    if (n == 0 || size < need) {
        return false;
    }
    out->header = header;
    out->meta = reinterpret_cast<const BlobBoardMeta *>(blob + sizeof(BlobHeader));
    out->images = blob + sizeof(BlobHeader) + n * sizeof(BlobBoardMeta);
    return true;
}

namespace {

void unpack_image(const uint8_t *packed, uint8_t *dst)
{
    const size_t n = kPackedImageBytes;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t byte = packed[i];
        uint8_t *d = dst + i * 8;
        d[0] = (byte >> 7) & 1u;
        d[1] = (byte >> 6) & 1u;
        d[2] = (byte >> 5) & 1u;
        d[3] = (byte >> 4) & 1u;
        d[4] = (byte >> 3) & 1u;
        d[5] = (byte >> 2) & 1u;
        d[6] = (byte >> 1) & 1u;
        d[7] = byte & 1u;
    }
}

}  // namespace

void unpack_board(const BoardSet &set, int index, const Board &board)
{
    const uint8_t *base = set.images + static_cast<size_t>(index) * 2 * kPackedImageBytes;
    unpack_image(base, board.test);
    unpack_image(base + kPackedImageBytes, board.temp);
}

void Score::add(const Score &o)
{
    n_defects += o.n_defects;
    n_found += o.n_found;
    n_unreachable += o.n_unreachable;
    n_lost_to_gate += o.n_lost_to_gate;
    true_positives += o.true_positives;
    false_positives += o.false_positives;
    ignored_firings += o.ignored_firings;
    clean_boards += o.clean_boards;
    clean_boards_with_alarm += o.clean_boards_with_alarm;
    parity_checked += o.parity_checked;
    parity_mismatch += o.parity_mismatch;
    for (int c = 0; c < kNumClasses; ++c) {
        per_class_defects[c] += o.per_class_defects[c];
        per_class_found[c] += o.per_class_found[c];
    }
}

Score score_board(const BlobBoardMeta &meta, const BoardOutput &out)
{
    Score s;

    // Defect-level recall: found if any qualifying tile fired its class.
    for (uint32_t d = 0; d < meta.n_defects && d < kMaxDefects; ++d) {
        const BlobDefect &def = meta.defects[d];
        s.n_defects++;
        s.per_class_defects[def.cls]++;
        if (def.tile_mask == 0) {
            s.n_unreachable++;   // geometry, not the model
            continue;
        }
        bool found = false;
        bool any_ran = false;
        for (int t = 0; t < kTilesPerBoard; ++t) {
            if (!(def.tile_mask & (1u << t))) {
                continue;
            }
            any_ran |= out.ran[t];
            if (out.ran[t] && (out.fired[t] & (1u << def.cls))) {
                found = true;
            }
        }
        if (found) {
            s.n_found++;
            s.per_class_found[def.cls]++;
        } else if (!any_ran) {
            s.n_lost_to_gate++;
        }
    }

    // (tile, class) precision, and parity with the PC int8 decisions on the
    // tiles that actually ran.
    bool any_alarm = false;
    for (int t = 0; t < kTilesPerBoard; ++t) {
        if (!out.ran[t]) {
            continue;
        }
        for (int c = 0; c < kNumClasses; ++c) {
            const uint8_t bit = static_cast<uint8_t>(1u << c);
            const bool fired = out.fired[t] & bit;
            const bool positive = meta.y_mask[t] & bit;
            const bool ignored = meta.ignore_mask[t] & bit;
            if (fired) {
                any_alarm = true;
                if (positive) {
                    s.true_positives++;
                } else if (ignored) {
                    s.ignored_firings++;
                } else {
                    s.false_positives++;
                }
            }
            if (meta.has_expect) {
                s.parity_checked++;
                if (fired != static_cast<bool>(meta.expect_mask[t] & bit)) {
                    s.parity_mismatch++;
                }
            }
        }
    }
    if (meta.is_clean) {
        s.clean_boards = 1;
        s.clean_boards_with_alarm = any_alarm ? 1 : 0;
    }
    return s;
}

}  // namespace pcb
