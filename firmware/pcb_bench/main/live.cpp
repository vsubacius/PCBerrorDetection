#include "live.hpp"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace pcb {
namespace live {

namespace {

void (*g_yield)() = nullptr;

inline void yield_now()
{
    if (g_yield != nullptr) {
        g_yield();
    }
}

// The ESP32-S3 FPU is single precision only; double math runs in software and
// is very slow. Hot loops use this float copy of the homography.
struct HomographyF {
    float h[9];
    explicit HomographyF(const double d[9])
    {
        for (int i = 0; i < 9; ++i) {
            h[i] = static_cast<float>(d[i]);
        }
    }
    Point map(float x, float y) const
    {
        const float inv = 1.0f / (h[6] * x + h[7] * y + h[8]);
        return {(h[0] * x + h[1] * y + h[2]) * inv, (h[3] * x + h[4] * y + h[5]) * inv};
    }
};

// --------------------------------------------------------------- blobs ----

struct Blob {
    int area = 0;
    double sum_x = 0;
    double sum_y = 0;
    int min_x = 1 << 30;
    int min_y = 1 << 30;
    int max_x = -1;
    int max_y = -1;
    bool touches_border = false;
};

struct Run {
    int y;
    int x0;
    int x1;   // inclusive
    int parent;
};

int find_root(std::vector<Run> &runs, int i)
{
    while (runs[i].parent != i) {
        runs[i].parent = runs[runs[i].parent].parent;
        i = runs[i].parent;
    }
    return i;
}

void unite(std::vector<Run> &runs, int a, int b)
{
    a = find_root(runs, a);
    b = find_root(runs, b);
    if (a != b) {
        if (a < b) {
            runs[b].parent = a;
        } else {
            runs[a].parent = b;
        }
    }
}

// 8-connected dark (gray <= threshold) blobs, by run-length union-find.
std::vector<Blob> dark_blobs(const Frame &f, uint8_t threshold)
{
    std::vector<Run> runs;
    runs.reserve(static_cast<size_t>(f.h) * 8);
    int prev_begin = 0;
    int prev_end = 0;
    for (int y = 0; y < f.h; ++y) {
        const uint8_t *row = f.gray + static_cast<size_t>(y) * f.w;
        const int begin = static_cast<int>(runs.size());
        int x = 0;
        while (x < f.w) {
            while (x < f.w && row[x] > threshold) {
                ++x;
            }
            if (x >= f.w) {
                break;
            }
            const int x0 = x;
            while (x < f.w && row[x] <= threshold) {
                ++x;
            }
            const int id = static_cast<int>(runs.size());
            runs.push_back({y, x0, x - 1, id});
            // 8-connectivity: overlap with the previous row's runs, widened by 1.
            for (int p = prev_begin; p < prev_end; ++p) {
                if (runs[p].x1 >= x0 - 1 && runs[p].x0 <= x) {
                    unite(runs, id, p);
                }
            }
        }
        prev_begin = begin;
        prev_end = static_cast<int>(runs.size());
    }

    std::vector<int> blob_of(runs.size(), -1);
    std::vector<Blob> blobs;
    for (size_t i = 0; i < runs.size(); ++i) {
        const int root = find_root(runs, static_cast<int>(i));
        if (blob_of[root] < 0) {
            blob_of[root] = static_cast<int>(blobs.size());
            blobs.emplace_back();
        }
        Blob &b = blobs[blob_of[root]];
        const Run &r = runs[i];
        const int n = r.x1 - r.x0 + 1;
        b.area += n;
        b.sum_x += (static_cast<double>(r.x0) + r.x1) * 0.5 * n;
        b.sum_y += static_cast<double>(r.y) * n;
        b.min_x = std::min(b.min_x, r.x0);
        b.max_x = std::max(b.max_x, r.x1);
        b.min_y = std::min(b.min_y, r.y);
        b.max_y = std::max(b.max_y, r.y);
        if (r.x0 == 0 || r.x1 == f.w - 1 || r.y == 0 || r.y == f.h - 1) {
            b.touches_border = true;
        }
    }
    return blobs;
}

// ---------------------------------------------------------- sampling ----

inline float bilinear(const Frame &f, float x, float y)
{
    if (x < 0 || y < 0 || x > f.w - 1.001f || y > f.h - 1.001f) {
        return 255.0f;   // outside the frame: treat as paper
    }
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const float ax = x - x0;
    const float ay = y - y0;
    const uint8_t *p = f.gray + static_cast<size_t>(y0) * f.w + x0;
    const float top = p[0] + ax * (p[1] - p[0]);
    const float bottom = p[f.w] + ax * (p[f.w + 1] - p[f.w]);
    return top + ay * (bottom - top);
}

// Board-pixel position of the fiducial at board corner c (TL, TR, BR, BL).
Point fiducial_board_position(int c)
{
    const float lo = -kFidOffset;
    const float hi = kBoardPx + kFidOffset;
    switch (c) {
    case 0: return {lo, lo};
    case 1: return {hi, lo};
    case 2: return {hi, hi};
    default: return {lo, hi};
    }
}

float distance(const Point &a, const Point &b)
{
    return hypotf(a.x - b.x, a.y - b.y);
}

}  // namespace

// ------------------------------------------------------------ public ----

void set_yield(void (*fn)())
{
    g_yield = fn;
}

uint8_t otsu_threshold(const uint32_t hist[256])
{
    double total = 0;
    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        total += hist[i];
        sum += static_cast<double>(i) * hist[i];
    }
    double w0 = 0;
    double sum0 = 0;
    double best = -1;
    int best_t = 127;
    for (int t = 0; t < 256; ++t) {
        w0 += hist[t];
        if (w0 == 0) {
            continue;
        }
        const double w1 = total - w0;
        if (w1 == 0) {
            break;
        }
        sum0 += static_cast<double>(t) * hist[t];
        const double m0 = sum0 / w0;
        const double m1 = (sum - sum0) / w1;
        const double between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (between > best) {
            best = between;
            best_t = t;
        }
    }
    return static_cast<uint8_t>(best_t);
}

void build_flat_field(const Frame &blank, FlatField *ff)
{
    ff->w = blank.w;
    ff->h = blank.h;
    ff->cw = (blank.w + ff->cell - 1) / ff->cell;
    ff->ch = (blank.h + ff->cell - 1) / ff->cell;
    for (int cy = 0; cy < ff->ch; ++cy) {
        for (int cx = 0; cx < ff->cw; ++cx) {
            uint32_t sum = 0;
            uint32_t n = 0;
            for (int y = cy * ff->cell; y < std::min(blank.h, (cy + 1) * ff->cell); ++y) {
                for (int x = cx * ff->cell; x < std::min(blank.w, (cx + 1) * ff->cell); ++x) {
                    sum += blank.gray[static_cast<size_t>(y) * blank.w + x];
                    ++n;
                }
            }
            ff->mean[cy * ff->cw + cx] = static_cast<uint16_t>(std::max<uint32_t>(1, sum / n));
        }
    }
}

void build_auto_background(const Frame &frame, int spread, FlatField *ff)
{
    ff->w = frame.w;
    ff->h = frame.h;
    ff->cw = (frame.w + ff->cell - 1) / ff->cell;
    ff->ch = (frame.h + ff->cell - 1) / ff->cell;
    std::vector<uint16_t> cell_max(static_cast<size_t>(ff->cw) * ff->ch, 1);
    for (int y = 0; y < frame.h; ++y) {
        const uint8_t *row = frame.gray + static_cast<size_t>(y) * frame.w;
        uint16_t *cm = cell_max.data() + static_cast<size_t>(y / ff->cell) * ff->cw;
        for (int x = 0; x < frame.w; ++x) {
            uint16_t &m = cm[x / ff->cell];
            if (row[x] > m) {
                m = row[x];
            }
        }
    }
    for (int cy = 0; cy < ff->ch; ++cy) {
        for (int cx = 0; cx < ff->cw; ++cx) {
            uint16_t m = 1;
            for (int dy = -spread; dy <= spread; ++dy) {
                const int yy = cy + dy;
                if (yy < 0 || yy >= ff->ch) {
                    continue;
                }
                for (int dx = -spread; dx <= spread; ++dx) {
                    const int xx = cx + dx;
                    if (xx >= 0 && xx < ff->cw) {
                        m = std::max(m, cell_max[static_cast<size_t>(yy) * ff->cw + xx]);
                    }
                }
            }
            ff->mean[cy * ff->cw + cx] = m;
        }
    }
}

void apply_flat_field(uint8_t *gray, int w, int h, const FlatField &ff)
{
    if (ff.mean == nullptr || ff.w != w || ff.h != h) {
        return;
    }
    // gray * 230 / background, as a gain per cell in 8.8 fixed point,
    // interpolated bilinearly between cell centres. No division per pixel.
    std::vector<int32_t> gain(static_cast<size_t>(ff.cw) * ff.ch);
    for (int i = 0; i < ff.cw * ff.ch; ++i) {
        gain[i] = (230 << 8) / std::max<int>(1, ff.mean[i]);
    }
    std::vector<int32_t> row_gain(ff.cw);
    // Per-pixel x interpolation table: left cell and weight (0..256).
    std::vector<int16_t> x_cell(w);
    std::vector<int16_t> x_weight(w);
    for (int x = 0; x < w; ++x) {
        float fx = (x + 0.5f) / ff.cell - 0.5f;
        fx = std::max(0.0f, std::min(fx, static_cast<float>(ff.cw - 1)));
        const int x0 = std::min(static_cast<int>(fx), std::max(0, ff.cw - 2));
        x_cell[x] = static_cast<int16_t>(x0);
        x_weight[x] = static_cast<int16_t>(ff.cw > 1 ? (fx - x0) * 256.0f + 0.5f : 0);
    }
    for (int y = 0; y < h; ++y) {
        float fy = (y + 0.5f) / ff.cell - 0.5f;
        fy = std::max(0.0f, std::min(fy, static_cast<float>(ff.ch - 1)));
        const int y0 = static_cast<int>(fy);
        const int y1 = std::min(y0 + 1, ff.ch - 1);
        const int ay = static_cast<int>((fy - y0) * 256.0f + 0.5f);
        for (int cx = 0; cx < ff.cw; ++cx) {
            row_gain[cx] = (gain[y0 * ff.cw + cx] * (256 - ay) + gain[y1 * ff.cw + cx] * ay) >> 8;
        }
        uint8_t *row = gray + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const int c = x_cell[x];
            const int a = x_weight[x];
            const int32_t g1 = row_gain[std::min(c + 1, ff.cw - 1)];
            const int32_t g = (row_gain[c] * (256 - a) + g1 * a) >> 8;
            const int32_t v = (row[x] * g) >> 8;
            row[x] = static_cast<uint8_t>(v > 255 ? 255 : v);
        }
        if ((y & 63) == 63) {
            yield_now();
        }
    }
}

bool solve_homography(const Point src[4], const Point dst[4], double h[9])
{
    // A * [h0..h7] = b, h8 = 1.
    double a[8][9];
    for (int i = 0; i < 4; ++i) {
        const double x = src[i].x;
        const double y = src[i].y;
        const double u = dst[i].x;
        const double v = dst[i].y;
        double *r0 = a[2 * i];
        double *r1 = a[2 * i + 1];
        const double row0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
        const double row1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        memcpy(r0, row0, sizeof(row0));
        memcpy(r1, row1, sizeof(row1));
    }
    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 8; ++r) {
            if (fabs(a[r][col]) > fabs(a[pivot][col])) {
                pivot = r;
            }
        }
        if (fabs(a[pivot][col]) < 1e-12) {
            return false;
        }
        if (pivot != col) {
            for (int k = 0; k < 9; ++k) {
                std::swap(a[col][k], a[pivot][k]);
            }
        }
        for (int r = 0; r < 8; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = a[r][col] / a[col][col];
            for (int k = col; k < 9; ++k) {
                a[r][k] -= factor * a[col][k];
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        h[i] = a[i][8] / a[i][i];
    }
    h[8] = 1.0;
    return true;
}

Point apply_homography(const double h[9], float x, float y)
{
    const double w = h[6] * x + h[7] * y + h[8];
    return {static_cast<float>((h[0] * x + h[1] * y + h[2]) / w),
            static_cast<float>((h[3] * x + h[4] * y + h[5]) / w)};
}

AlignResult align_frame(const Frame &frame, const uint8_t *template_board,
                        uint8_t *out_test, uint8_t *warp_scratch)
{
    AlignResult r;

    // 1. Frame threshold.
    uint32_t hist[256] = {};
    for (size_t i = 0, n = static_cast<size_t>(frame.w) * frame.h; i < n; ++i) {
        hist[frame.gray[i]]++;
    }
    r.frame_threshold = otsu_threshold(hist);

    // 2. Dark blobs, and the ones that look like a filled square.
    yield_now();
    const std::vector<Blob> blobs = dark_blobs(frame, r.frame_threshold);
    yield_now();
    r.n_blobs = static_cast<int>(blobs.size());
    const float frame_area = static_cast<float>(frame.w) * frame.h;
    std::vector<int> cand;
    for (size_t i = 0; i < blobs.size(); ++i) {
        const Blob &b = blobs[i];
        if (b.touches_border) {
            continue;
        }
        const float bw = static_cast<float>(b.max_x - b.min_x + 1);
        const float bh = static_cast<float>(b.max_y - b.min_y + 1);
        const float fill = b.area / (bw * bh);
        const float aspect = bw / bh;
        if (b.area < 0.00015f * frame_area || b.area > 0.03f * frame_area) {
            continue;
        }
        if (fill < 0.5f || aspect < 0.5f || aspect > 2.0f) {
            continue;
        }
        cand.push_back(static_cast<int>(i));
    }
    r.n_candidates = static_cast<int>(cand.size());
    if (cand.size() < 4) {
        snprintf(r.message, sizeof(r.message),
                 "only %d square-ish dark blobs found (need the 4 corner squares); "
                 "check that the whole sheet pattern is in view", r.n_candidates);
        return r;
    }

    // 3. Extreme candidates in the four diagonal directions (image TL, TR, BR, BL).
    const int dir_x[4] = {-1, 1, 1, -1};
    const int dir_y[4] = {-1, -1, 1, 1};
    int pick[4];
    for (int d = 0; d < 4; ++d) {
        double best = -1e30;
        pick[d] = -1;
        for (int i : cand) {
            const Blob &b = blobs[i];
            const double s = dir_x[d] * b.sum_x / b.area + dir_y[d] * b.sum_y / b.area;
            if (s > best) {
                best = s;
                pick[d] = i;
            }
        }
    }
    Point img_pt[4];
    float img_area[4];
    for (int d = 0; d < 4; ++d) {
        for (int e = 0; e < d; ++e) {
            if (pick[d] == pick[e]) {
                snprintf(r.message, sizeof(r.message),
                         "corner squares not found: one blob is extreme in two directions "
                         "(sheet rotated more than ~30 degrees?)");
                return r;
            }
        }
        const Blob &b = blobs[pick[d]];
        img_pt[d] = {static_cast<float>(b.sum_x / b.area), static_cast<float>(b.sum_y / b.area)};
        img_area[d] = static_cast<float>(b.area);
    }

    // Geometry check: a near-square quadrilateral with similar blobs.
    float side[4];
    for (int d = 0; d < 4; ++d) {
        side[d] = distance(img_pt[d], img_pt[(d + 1) % 4]);
    }
    const float side_min = *std::min_element(side, side + 4);
    const float side_max = *std::max_element(side, side + 4);
    const float diag_ratio = distance(img_pt[0], img_pt[2]) / std::max(1.0f, distance(img_pt[1], img_pt[3]));
    const float area_min = *std::min_element(img_area, img_area + 4);
    const float area_max = *std::max_element(img_area, img_area + 4);
    const float mean_side = (side[0] + side[1] + side[2] + side[3]) / 4;
    const float size_ratio = sqrtf((area_min + area_max) / 2) / mean_side;
    const float expected_ratio = kFidSize / (kBoardPx + 2 * kFidOffset);
    if (side_max > 1.5f * side_min || diag_ratio > 1.3f || diag_ratio < 1 / 1.3f ||
        area_max > 2.5f * area_min || size_ratio < 0.5f * expected_ratio ||
        size_ratio > 2.0f * expected_ratio) {
        snprintf(r.message, sizeof(r.message),
                 "corner squares failed the shape check (sides %.0f..%.0f px, diagonal ratio "
                 "%.2f, blob areas %.0f..%.0f, size ratio %.3f vs %.3f)",
                 side_min, side_max, diag_ratio, area_min, area_max, size_ratio, expected_ratio);
        return r;
    }

    // 4. Try the 8 ways to match image corners to board corners.
    //    Image picks are in clockwise order (TL, TR, BR, BL in image coords).
    //    Board corners are also clockwise in board coords (y down). A mirror
    //    reverses the order.
    constexpr int kCoarse = 64;   // coarse grid per side
    r.coarse_samples = kCoarse * kCoarse;
    int best_mis = 1 << 30;
    int second_mis = 1 << 30;
    double best_h[9] = {};
    int best_o = -1;
    for (int o = 0; o < 8; ++o) {
        const int rot = o & 3;
        const bool mirror = o >= 4;
        Point src[4];
        for (int d = 0; d < 4; ++d) {
            const int corner = mirror ? ((rot - d) % 4 + 4) % 4 : (rot + d) % 4;
            src[d] = fiducial_board_position(corner);
        }
        double h[9];
        if (!solve_homography(src, img_pt, h)) {
            continue;
        }
        const HomographyF hf(h);
        int mis = 0;
        for (int gy = 0; gy < kCoarse; ++gy) {
            const int by = gy * kBoardPx / kCoarse + kBoardPx / (2 * kCoarse);
            for (int gx = 0; gx < kCoarse; ++gx) {
                const int bx = gx * kBoardPx / kCoarse + kBoardPx / (2 * kCoarse);
                const Point p = hf.map(static_cast<float>(bx), static_cast<float>(by));
                const uint8_t bit = bilinear(frame, p.x, p.y) > r.frame_threshold ? 1 : 0;
                mis += bit != template_board[by * kBoardPx + bx];
            }
        }
        if (mis < best_mis) {
            second_mis = best_mis;
            best_mis = mis;
            best_o = o;
            memcpy(best_h, h, sizeof(h));
        } else if (mis < second_mis) {
            second_mis = mis;
        }
    }
    if (best_o < 0) {
        snprintf(r.message, sizeof(r.message), "homography failed for every orientation");
        return r;
    }
    // A real match disagrees with the template on a few percent of samples
    // (defects, print and camera error); a wrong one on a third or more.
    if (best_mis > r.coarse_samples * 15 / 100) {
        snprintf(r.message, sizeof(r.message),
                 "the printed board does not match the stored template (best %d of %d samples "
                 "differ, need < 15%%): wrong board number selected, or a corner square not in view",
                 best_mis, r.coarse_samples);
        r.orientation = best_o;
        r.coarse_mismatch = best_mis;
        r.coarse_second = second_mis;
        return r;
    }
    r.orientation = best_o;
    r.coarse_mismatch = best_mis;
    r.coarse_second = second_mis;
    memcpy(r.homography, best_h, sizeof(best_h));
    for (int c = 0; c < 4; ++c) {
        const Point fp = fiducial_board_position(c);
        const Point ip = apply_homography(best_h, fp.x, fp.y);
        // The image blob nearest to where board corner c's fiducial maps.
        int nearest = 0;
        for (int d = 1; d < 4; ++d) {
            if (distance(img_pt[d], ip) < distance(img_pt[nearest], ip)) {
                nearest = d;
            }
        }
        r.fiducial[c] = img_pt[nearest];
        r.fiducial_area[c] = img_area[nearest];
    }
    r.board_side_px = mean_side * kBoardPx / (kBoardPx + 2 * kFidOffset);

    // 5. Resample the board with a margin, bilinear. Float, with the
    //    homography terms stepped along each row (no double math per pixel).
    yield_now();
    {
        const HomographyF hf(best_h);
        const float *m = hf.h;
        for (int wy = 0; wy < kWarpSize; ++wy) {
            const float by = static_cast<float>(wy - kShiftRadius);
            const float bx0 = static_cast<float>(-kShiftRadius);
            float nu = m[0] * bx0 + m[1] * by + m[2];
            float nv = m[3] * bx0 + m[4] * by + m[5];
            float dw = m[6] * bx0 + m[7] * by + m[8];
            uint8_t *row = warp_scratch + static_cast<size_t>(wy) * kWarpSize;
            for (int wx = 0; wx < kWarpSize; ++wx) {
                const float inv = 1.0f / dw;
                const float v = bilinear(frame, nu * inv, nv * inv);
                row[wx] = static_cast<uint8_t>(v + 0.5f);
                nu += m[0];
                nv += m[3];
                dw += m[6];
            }
            if ((wy & 63) == 63) {
                yield_now();
            }
        }
    }

    // 6. Board threshold from the board area only, and sharpness.
    uint32_t bhist[256] = {};
    double lap = 0;
    for (int y = kShiftRadius; y < kShiftRadius + kBoardPx; ++y) {
        const uint8_t *row = warp_scratch + static_cast<size_t>(y) * kWarpSize;
        for (int x = kShiftRadius; x < kShiftRadius + kBoardPx; ++x) {
            bhist[row[x]]++;
            lap += abs(4 * row[x] - row[x - 1] - row[x + 1] - row[x - kWarpSize] - row[x + kWarpSize]);
        }
    }
    r.board_threshold = otsu_threshold(bhist);
    r.sharpness = static_cast<float>(lap / (static_cast<double>(kBoardPx) * kBoardPx));
    const uint8_t thr = r.board_threshold;

    // 7. Shift search against the template. Threshold the resampled board once
    //    (in place: 0/1), then search all shifts on every 4th pixel and refine
    //    the best ones on every 2nd pixel.
    yield_now();
    for (size_t i = 0, n = static_cast<size_t>(kWarpSize) * kWarpSize; i < n; ++i) {
        warp_scratch[i] = warp_scratch[i] > thr ? 1 : 0;
    }
    auto mismatch = [&](int dx, int dy, int step) {
        int mis = 0;
        for (int y = 0; y < kBoardPx; y += step) {
            const uint8_t *w = warp_scratch + static_cast<size_t>(y + kShiftRadius + dy) * kWarpSize
                               + kShiftRadius + dx;
            const uint8_t *t = template_board + static_cast<size_t>(y) * kBoardPx;
            for (int x = 0; x < kBoardPx; x += step) {
                mis += w[x] ^ t[x];
            }
        }
        return mis;
    };
    // Coarse pass: keep the 4 best shifts.
    constexpr int kKeep = 4;
    int keep_mis[kKeep];
    int keep_dx[kKeep];
    int keep_dy[kKeep];
    for (int k = 0; k < kKeep; ++k) {
        keep_mis[k] = 1 << 30;
        keep_dx[k] = 0;
        keep_dy[k] = 0;
    }
    for (int dy = -kShiftRadius; dy <= kShiftRadius; ++dy) {
        for (int dx = -kShiftRadius; dx <= kShiftRadius; ++dx) {
            const int mis = mismatch(dx, dy, 4);
            int worst = 0;
            for (int k = 1; k < kKeep; ++k) {
                if (keep_mis[k] > keep_mis[worst]) {
                    worst = k;
                }
            }
            if (mis < keep_mis[worst]) {
                keep_mis[worst] = mis;
                keep_dx[worst] = dx;
                keep_dy[worst] = dy;
            }
        }
        yield_now();
    }
    // Fine pass: each kept shift and its 8 neighbours, on every 2nd pixel.
    int best_shift_mis = 1 << 30;
    r.shift_samples = (kBoardPx / 2) * (kBoardPx / 2);
    for (int k = 0; k < kKeep; ++k) {
        for (int ny = -1; ny <= 1; ++ny) {
            for (int nx = -1; nx <= 1; ++nx) {
                const int dx = keep_dx[k] + nx;
                const int dy = keep_dy[k] + ny;
                if (abs(dx) > kShiftRadius || abs(dy) > kShiftRadius) {
                    continue;
                }
                const int mis = mismatch(dx, dy, 2);
                // Prefer the smaller shift on ties.
                if (mis < best_shift_mis ||
                    (mis == best_shift_mis && abs(dx) + abs(dy) < abs(r.shift_x) + abs(r.shift_y))) {
                    best_shift_mis = mis;
                    r.shift_x = dx;
                    r.shift_y = dy;
                }
            }
        }
    }
    r.shift_mismatch = best_shift_mis;

    for (int y = 0; y < kBoardPx; ++y) {
        const uint8_t *w = warp_scratch + static_cast<size_t>(y + kShiftRadius + r.shift_y) * kWarpSize
                           + kShiftRadius + r.shift_x;
        memcpy(out_test + static_cast<size_t>(y) * kBoardPx, w, kBoardPx);
    }
    r.ok = true;
    snprintf(r.message, sizeof(r.message), "aligned");
    return r;
}

}  // namespace live
}  // namespace pcb
