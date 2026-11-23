// Real DeepPCB validation boards embedded in flash (see tools/make_boards.py),
// plus the defect-level scoring that eval/metrics.py:evaluate_defects defines.
//
// The blob is written by make_boards.py; the layout here must match it
// byte for byte, which static_asserts and kMagic check.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "board.hpp"

namespace pcb {

constexpr int kNumClasses = 6;
constexpr int kMaxDefects = 48;

#pragma pack(push, 1)
struct BlobHeader {
    char magic[4];             // "PCB1"
    uint32_t version;          // 1
    uint32_t n_boards;
    uint32_t in_channels;      // 1 (mode A) or 2 (modes B, C)
    float thresholds_logit[kNumClasses];
    uint32_t representation;   // 'A', 'B' or 'C'
    uint8_t pad[64 - 4 - 4 * 4 - 4 * kNumClasses];
};

struct BlobDefect {
    uint8_t cls;               // class channel 0..5
    uint8_t pad[3];
    uint32_t tile_mask;        // bit t set: tile t may be credited (70% rule)
};

struct BlobBoardMeta {
    char name[16];
    uint32_t is_clean;         // 1: template used as test, every firing is false
    uint32_t n_defects;
    uint8_t y_mask[kTilesPerBoard];       // bit c: tile positive for class c
    uint8_t ignore_mask[kTilesPerBoard];  // bit c: (tile, c) ignored
    uint8_t expect_mask[kTilesPerBoard];  // bit c: PC int8 fires (gate off)
    uint8_t has_expect;
    BlobDefect defects[kMaxDefects];
    uint8_t pad[512 - 16 - 8 - 3 * kTilesPerBoard - 1 - 8 * kMaxDefects];
};
#pragma pack(pop)

static_assert(sizeof(BlobHeader) == 64, "header layout must match make_boards.py");
static_assert(sizeof(BlobDefect) == 8, "defect layout must match make_boards.py");
static_assert(sizeof(BlobBoardMeta) == 512, "meta layout must match make_boards.py");

constexpr size_t kPackedImageBytes = static_cast<size_t>(kBoardSize) * kBoardSize / 8;

struct BoardSet {
    const BlobHeader *header = nullptr;
    const BlobBoardMeta *meta = nullptr;   // n_boards entries
    const uint8_t *images = nullptr;       // n_boards * 2 packed images
};

// Validate and index a blob. Returns false (and leaves `out` empty) if the
// magic, version or size is wrong.
bool parse_board_blob(const uint8_t *blob, size_t size, BoardSet *out);

// Unpack board `index` (MSB-first bits) into a 0/1 byte board in PSRAM.
void unpack_board(const BoardSet &set, int index, const Board &board);

// What one pass over a board produced. fired[t] has bit c set when tile t
// fired class c; ran[t] is false for tiles the gate skipped.
struct BoardOutput {
    uint8_t fired[kTilesPerBoard] = {};
    bool ran[kTilesPerBoard] = {};
};

// Defect-level and (tile, class) counts, identical in definition to
// eval/metrics.py:evaluate_defects (match="any").
struct Score {
    int n_defects = 0;
    int n_found = 0;
    int n_unreachable = 0;
    int n_lost_to_gate = 0;
    int true_positives = 0;
    int false_positives = 0;
    int ignored_firings = 0;
    int clean_boards = 0;
    int clean_boards_with_alarm = 0;
    int parity_checked = 0;     // (tile, class) decisions compared with PC
    int parity_mismatch = 0;
    int per_class_defects[kNumClasses] = {};
    int per_class_found[kNumClasses] = {};

    void add(const Score &o);
};

Score score_board(const BlobBoardMeta &meta, const BoardOutput &out);

}  // namespace pcb
