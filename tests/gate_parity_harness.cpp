// Test harness: runs the firmware's Stage-1 gate on tiles fed in over stdin so
// tests/test_gate_parity.py can compare it against the numpy reference.
// Built on the host with g++; never compiled into the ESP-IDF app.
//
// stdin  : int32 tile, int32 radius, int32 n_pairs,
//          then n_pairs * (tile*tile) bytes of test tiles,
//          then n_pairs * (tile*tile) bytes of template tiles
// stdout : one int32 score per pair
#include <cstdio>
#include <cstdint>
#include <vector>

#include "gate.hpp"

static bool read_exact(void *dst, size_t n)
{
    return std::fread(dst, 1, n, stdin) == n;
}

int main()
{
    int32_t tile = 0, radius = 0, pairs = 0;
    if (!read_exact(&tile, 4) || !read_exact(&radius, 4) || !read_exact(&pairs, 4)) {
        std::fprintf(stderr, "short header\n");
        return 1;
    }
    const size_t plane = static_cast<size_t>(tile) * static_cast<size_t>(tile);
    std::vector<uint8_t> test(plane * pairs), temp(plane * pairs);
    if (!read_exact(test.data(), test.size()) || !read_exact(temp.data(), temp.size())) {
        std::fprintf(stderr, "short payload\n");
        return 1;
    }

    std::vector<uint8_t> scratch(pcb::gate_scratch_bytes(tile, radius));
    for (int32_t i = 0; i < pairs; ++i) {
        const int32_t score = pcb::gate_score(
            test.data() + plane * i, temp.data() + plane * i,
            tile, tile, radius, scratch.data());
        std::fwrite(&score, 4, 1, stdout);
    }
    return 0;
}
