// Live mode: turn one camera frame of a printed board sheet into a 640x640
// binary test image aligned to the stored template.
//
// The sheet (tools/make_prints.py) has the board image plus four solid black
// squares ("fiducials"), one diagonally outside each board corner. Steps:
//
//   1. optional flat-field correction (divide by a blank-paper frame);
//   2. Otsu threshold of the frame; dark connected blobs;
//   3. the four fiducials are the filled, square-ish blobs that are extreme in
//      the four diagonal directions; their geometry is checked;
//   4. the frame may be rotated by any multiple of 90 degrees or mirrored, so
//      all 8 ways of matching the four blobs to the four board corners are
//      tried on a coarse grid against the template, and the best one is kept;
//   5. a homography maps board pixels to camera pixels; the board is
//      resampled (bilinear) with a margin of kShiftRadius pixels;
//   6. Otsu on the resampled board gives the black/white threshold;
//   7. a +-kShiftRadius pixel shift search against the template removes the
//      last error left by the fiducial centres.
//
// Pixel value 1 is white (paper), 0 is black (ink), the same as DeepPCB.
//
// No ESP-IDF code here, so the whole file also builds and is tested on a PC.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace pcb {
namespace live {

constexpr int kBoardPx = 640;
// Fiducial geometry in board pixels; must match tools/make_prints.py.
constexpr float kFidSize = 56.0f;     // side of each black square
constexpr float kFidOffset = 60.0f;   // centre is this far outside the corner, in x and in y
constexpr int kShiftRadius = 6;       // shift search range, board pixels
constexpr int kWarpSize = kBoardPx + 2 * kShiftRadius;

struct Frame {
    const uint8_t *gray;   // row-major, w*h bytes
    int w;
    int h;
};

struct Point {
    float x;
    float y;
};

struct AlignResult {
    bool ok = false;
    char message[224] = {};

    int n_blobs = 0;            // dark blobs in the frame
    int n_candidates = 0;       // blobs that look like a filled square
    uint8_t frame_threshold = 0;
    uint8_t board_threshold = 0;

    Point fiducial[4] = {};     // camera position of the fiducial at board TL, TR, BR, BL
    float fiducial_area[4] = {};
    float board_side_px = 0;    // board side length in camera pixels (mean)
    double homography[9] = {};  // board pixel -> camera pixel

    int orientation = -1;       // 0..7: rotation (0..3) + 4 * mirrored
    int coarse_mismatch = 0;    // best orientation, of coarse_samples
    int coarse_second = 0;      // second-best orientation
    int coarse_samples = 0;
    int expected_board = -1;    // set by the caller, for messages only

    int shift_x = 0;            // applied shift, board pixels
    int shift_y = 0;
    int shift_mismatch = 0;     // of shift_samples, at the best shift
    int shift_samples = 0;

    float sharpness = 0;        // mean |Laplacian| of the resampled board (relative)
};

// Flat field: per-cell mean of a blank white sheet, applied as gray * 230 / bg.
struct FlatField {
    int w = 0;
    int h = 0;
    int cell = 16;
    int cw = 0;
    int ch = 0;
    uint16_t *mean = nullptr;   // cw*ch values, caller-owned storage
};

// Fill `ff->mean` from a blank-paper frame. `ff->mean` must hold
// ceil(w/cell)*ceil(h/cell) values.
void build_flat_field(const Frame &blank, FlatField *ff);
// In-place correction of a frame with the same size as the flat field.
void apply_flat_field(uint8_t *gray, int w, int h, const FlatField &ff);

// Without a recorded flat field: estimate the paper brightness from the frame
// itself (per-cell maximum, spread over +-`spread` cells so large black copper
// areas still see paper) and divide by it. Corrects vignetting and uneven
// light. `ff->mean` must hold ceil(w/cell)*ceil(h/cell) values.
void build_auto_background(const Frame &frame, int spread, FlatField *ff);

uint8_t otsu_threshold(const uint32_t hist[256]);

// Full pipeline. `template_board` is the stored 640x640 0/1 template.
// `out_test` receives the 640x640 0/1 aligned test image.
// `warp_scratch` must hold kWarpSize*kWarpSize bytes (resampled gray).
AlignResult align_frame(const Frame &frame, const uint8_t *template_board,
                        uint8_t *out_test, uint8_t *warp_scratch);

// Optional: called between long steps and every few rows, so the firmware can
// let other tasks run (the task watchdog). Not needed on a PC.
void set_yield(void (*fn)());

// Board-pixel -> camera-pixel mapping with a homography.
Point apply_homography(const double h[9], float x, float y);

// 4-point homography: maps src[i] to dst[i]. Returns false if singular.
bool solve_homography(const Point src[4], const Point dst[4], double h[9]);

}  // namespace live
}  // namespace pcb
