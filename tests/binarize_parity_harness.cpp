// Test harness: runs the firmware's binarization on frames fed in over stdin
// so tests/test_binarize_parity.py can compare it against data/binarize.py.
// Built on the host with g++; never compiled into the ESP-IDF app.
//
// stdin  : int32 width, int32 height, int32 n_frames,
//          int32 flat_field (0/1), int32 invert (0/1), int32 threshold (<0 = Otsu),
//          then n_frames * (width*height) bytes of single-channel frames
// stdout : per frame, int32 threshold used then width*height bytes of {0,1}
#include <cstdio>
#include <cstdint>
#include <vector>

#include "pcb_binarize.hpp"

static bool read_exact(void *dst, size_t n)
{
    return std::fread(dst, 1, n, stdin) == n;
}

int main()
{
    int32_t width = 0, height = 0, frames = 0, flat = 0, invert = 0, threshold = -1;
    if (!read_exact(&width, 4) || !read_exact(&height, 4) || !read_exact(&frames, 4)
        || !read_exact(&flat, 4) || !read_exact(&invert, 4) || !read_exact(&threshold, 4)) {
        std::fprintf(stderr, "short header\n");
        return 1;
    }

    const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> input(plane * static_cast<size_t>(frames));
    if (!read_exact(input.data(), input.size())) {
        std::fprintf(stderr, "short payload\n");
        return 1;
    }

    std::vector<uint8_t> out(plane);
    std::vector<uint8_t> scratch(pcb::binarize_scratch_bytes(width, height, flat != 0) + 8);
    for (int32_t i = 0; i < frames; ++i) {
        const int32_t used = pcb::binarize(
            input.data() + plane * static_cast<size_t>(i), width, height, width,
            flat != 0, invert != 0, threshold, out.data(), scratch.data());
        std::fwrite(&used, 4, 1, stdout);
        std::fwrite(out.data(), 1, out.size(), stdout);
    }
    return 0;
}
