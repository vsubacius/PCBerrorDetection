// Test harness: runs the firmware's binarization on frames fed in over stdin
// so tests/test_binarize_parity.py can compare it against data/binarize.py.
// Built on the host with g++; never compiled into the ESP-IDF app.
//
// Two modes:
//   mode 0 -- single channel: pcb::binarize
//   mode 1 -- board: pcb::binarize_board, two planes (red then green) per frame
//
// stdin  : int32 mode, int32 width, int32 height, int32 n_frames,
//          int32 flat_field (0/1), int32 invert (0/1),
//          int32 threshold / t_red (<0 = Otsu), int32 t_green (<0 = Otsu),
//          then the frames: mode 0 one plane each, mode 1 two planes each
// stdout : per frame, int32 threshold (mode 0) or int32 t_red + int32 t_green
//          (mode 1), then width*height bytes of {0,1}
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
    int32_t mode = 0, width = 0, height = 0, frames = 0, flat = 0, invert = 0;
    int32_t threshold = -1, t_green = -1;
    if (!read_exact(&mode, 4) || !read_exact(&width, 4) || !read_exact(&height, 4)
        || !read_exact(&frames, 4) || !read_exact(&flat, 4) || !read_exact(&invert, 4)
        || !read_exact(&threshold, 4) || !read_exact(&t_green, 4)) {
        std::fprintf(stderr, "short header\n");
        return 1;
    }

    const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t planes_per_frame = (mode == 1) ? 2u : 1u;
    std::vector<uint8_t> input(plane * planes_per_frame * static_cast<size_t>(frames));
    if (!read_exact(input.data(), input.size())) {
        std::fprintf(stderr, "short payload\n");
        return 1;
    }

    std::vector<uint8_t> out(plane);
    const size_t scratch_bytes =
        (mode == 1) ? pcb::binarize_board_scratch_bytes(width, height, flat != 0)
                    : pcb::binarize_scratch_bytes(width, height, flat != 0);
    std::vector<uint8_t> scratch(scratch_bytes + 8);

    for (int32_t i = 0; i < frames; ++i) {
        if (mode == 1) {
            const uint8_t *red = input.data() + plane * 2u * static_cast<size_t>(i);
            const uint8_t *green = red + plane;
            int used_red = 0, used_green = 0;
            pcb::binarize_board(red, green, width, height, width, flat != 0,
                                threshold, t_green, out.data(), scratch.data(),
                                &used_red, &used_green);
            const int32_t a = used_red, b = used_green;
            std::fwrite(&a, 4, 1, stdout);
            std::fwrite(&b, 4, 1, stdout);
        } else {
            const int32_t used = pcb::binarize(
                input.data() + plane * static_cast<size_t>(i), width, height, width,
                flat != 0, invert != 0, threshold, out.data(), scratch.data());
            std::fwrite(&used, 4, 1, stdout);
        }
        std::fwrite(out.data(), 1, out.size(), stdout);
    }
    return 0;
}
