#include "pcb_binarize.hpp"

namespace pcb {
namespace {

// floor(sqrt(v)) for a non-negative int64, exactly -- the bit-by-bit method,
// no floating point, so it cannot disagree with Python's math.isqrt.
int64_t isqrt64(int64_t v)
{
    if (v <= 0) {
        return 0;
    }
    uint64_t x = static_cast<uint64_t>(v);
    uint64_t bit = 1ull << 62;
    while (bit > x) {
        bit >>= 2;
    }
    uint64_t root = 0;
    while (bit != 0) {
        if (x >= root + bit) {
            x -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return static_cast<int64_t>(root);
}

inline int clamp_index(int i, int n)
{
    if (i < 0) {
        return 0;
    }
    if (i >= n) {
        return n - 1;
    }
    return i;
}

}  // namespace

void histogram(const uint8_t *gray, int width, int height, int stride,
               uint32_t hist[256])
{
    for (int i = 0; i < 256; ++i) {
        hist[i] = 0;
    }
    for (int y = 0; y < height; ++y) {
        const uint8_t *row = gray + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
            ++hist[row[x]];
        }
    }
}

int otsu_threshold(const uint32_t hist[256])
{
    int64_t total = 0;
    int64_t sum_total = 0;
    for (int i = 0; i < 256; ++i) {
        total += hist[i];
        sum_total += static_cast<int64_t>(i) * hist[i];
    }
    if (total == 0) {
        return 255;
    }

    int64_t w0 = 0;
    int64_t s0 = 0;
    int64_t best_score = -1;
    int best_t = 255;
    for (int t = 0; t < 256; ++t) {
        w0 += hist[t];
        s0 += static_cast<int64_t>(t) * hist[t];
        const int64_t w1 = total - w0;
        if (w0 == 0 || w1 == 0) {
            continue;
        }
        // A = w0 * w1 * (mean0 - mean1), held exactly; ordering by
        // |A| / sqrt(w0 * w1) is ordering by the between-class variance.
        int64_t a = s0 * w1 - (sum_total - s0) * w0;
        if (a < 0) {
            a = -a;
        }
        const int64_t score = (a << 8) / isqrt64((w0 * w1) << 16);
        if (score > best_score) {
            best_score = score;
            best_t = t;
        }
    }
    return best_t;
}

void box_blur(const uint8_t *src, int width, int height, int stride, int radius,
              uint8_t *dst, int32_t *col_sums)
{
    if (radius <= 0) {
        for (int y = 0; y < height; ++y) {
            const uint8_t *in = src + static_cast<size_t>(y) * stride;
            uint8_t *out = dst + static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                out[x] = in[x];
            }
        }
        return;
    }

    const int side = 2 * radius + 1;
    const int32_t area = static_cast<int32_t>(side) * static_cast<int32_t>(side);

    // Column sums for the window centred on row 0: rows -radius..+radius with
    // the index clamped, which replicates the edge row exactly as numpy's
    // pad(mode="edge") does.
    for (int x = 0; x < width; ++x) {
        col_sums[x] = 0;
    }
    for (int dy = -radius; dy <= radius; ++dy) {
        const uint8_t *row = src + static_cast<size_t>(clamp_index(dy, height)) * stride;
        for (int x = 0; x < width; ++x) {
            col_sums[x] += row[x];
        }
    }

    for (int y = 0; y < height; ++y) {
        // Horizontal pass: a sliding sum over the column sums, edges clamped.
        int64_t window = 0;
        for (int dx = -radius; dx <= radius; ++dx) {
            window += col_sums[clamp_index(dx, width)];
        }
        uint8_t *out = dst + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            out[x] = static_cast<uint8_t>(window / area);
            const int add = clamp_index(x + radius + 1, width);
            const int drop = clamp_index(x - radius, width);
            window += col_sums[add];
            window -= col_sums[drop];
        }

        if (y + 1 < height) {
            // Roll the column sums down one row.
            const uint8_t *drop_row =
                src + static_cast<size_t>(clamp_index(y - radius, height)) * stride;
            const uint8_t *add_row =
                src + static_cast<size_t>(clamp_index(y + radius + 1, height)) * stride;
            for (int x = 0; x < width; ++x) {
                col_sums[x] += static_cast<int32_t>(add_row[x]) - drop_row[x];
            }
        }
    }
}

void flat_field(const uint8_t *src, int width, int height, int stride,
                int radius, uint8_t *dst, int32_t *col_sums)
{
    box_blur(src, width, height, stride, radius, dst, col_sums);
    for (int y = 0; y < height; ++y) {
        const uint8_t *in = src + static_cast<size_t>(y) * stride;
        uint8_t *out = dst + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            int v = static_cast<int>(in[x]) - static_cast<int>(out[x]) + 128;
            if (v < 0) {
                v = 0;
            } else if (v > 255) {
                v = 255;
            }
            out[x] = static_cast<uint8_t>(v);
        }
    }
}

int binarize(const uint8_t *gray, int width, int height, int stride,
             bool use_flat_field, bool invert, int threshold,
             uint8_t *dst, uint8_t *scratch)
{
    const uint8_t *source = gray;
    int source_stride = stride;

    if (use_flat_field) {
        uint8_t *corrected = scratch;
        int32_t *col_sums = reinterpret_cast<int32_t *>(
            scratch + static_cast<size_t>(width) * static_cast<size_t>(height));
        flat_field(gray, width, height, stride, kFlatFieldRadius, corrected, col_sums);
        source = corrected;
        source_stride = width;
    }

    int t = threshold;
    if (t < 0) {
        uint32_t hist[256];
        histogram(source, width, height, source_stride, hist);
        t = otsu_threshold(hist);
    }

    for (int y = 0; y < height; ++y) {
        const uint8_t *in = source + static_cast<size_t>(y) * source_stride;
        uint8_t *out = dst + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const bool above = in[x] > t;
            out[x] = static_cast<uint8_t>(invert ? !above : above);
        }
    }
    return t;
}

}  // namespace pcb
