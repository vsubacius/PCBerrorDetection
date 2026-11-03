// Benchmark mode for the gated tile cascade on a Seeed XIAO ESP32-S3 Sense.
//
// Every number this project reports for the chip comes from here. The app:
//
//   1. records a memory baseline before anything is allocated;
//   2. loads the int8 .espdl student and runs dl::Model::test(), which compares
//      on-device output against the PC values ESP-PPQ embedded in the model --
//      this is the PC-vs-device parity check the experiment plan requires;
//   3. calls dl::Model::profile() for per-layer latency and a memory breakdown;
//   4. walks all 25 tiles of a board with the gate off and then on, timing
//      image access, gate, Stage-2 inference and postprocessing separately;
//   5. reports peak (minimum-free) internal SRAM and PSRAM.
//
// The gate is a *runtime* switch, deliberately: the experiment plan requires
// `reference` and `gate_only` to differ in nothing but the gate, so both rows
// must come from one firmware binary and one model file.
//
// Boards are synthetic until the microSD path lands (see board.hpp). Latency
// and memory are valid on synthetic boards; accuracy is not, because the gate
// skip rate depends on real defect statistics. Treat the skip counts printed
// here as a smoke test, not a result.

#include <inttypes.h>
#include <stdio.h>

#include "board.hpp"
#include "dl_model_base.hpp"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "gate.hpp"

namespace {

const char *TAG = "pcb_bench";

// Input channels the deployed model expects: 1 for input mode A (test tile
// only), 2 for B (test + template) and C (test + XOR). Must match the .espdl.
constexpr int kInputChannels = 1;

// Gate threshold on the eroded-XOR count. Selected on the validation fold in
// Python, never here; 5 is the published thr@98 for a 3x3 erosion at 0 px
// misalignment. The gate fires when score > threshold.
constexpr int kGateThreshold = 5;

// Per-class decision thresholds, in *logit* space. The model has no sigmoid,
// so a probability threshold p becomes log(p / (1 - p)); 0.0 here means p=0.5.
// Replace with the per-class values chosen on validation int8 outputs.
constexpr float kLogitThresholds[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
const char *kClassNames[6] = {"open", "short", "mousebite", "spur",
                              "copper", "pin-hole"};

// The model is embedded in .rodata by main/CMakeLists.txt. The symbol is
// "_binary_" + filename with dots replaced by underscores + "_start".
extern "C" const uint8_t model_espdl_start[] asm("_binary_student_c1_espdl_start");

struct StageTiming {
    int64_t fetch_us = 0;   // reaching the tile's pixels
    int64_t gate_us = 0;    // Stage-1 eroded-XOR gate
    int64_t fill_us = 0;    // quantising the tile into the model input
    int64_t infer_us = 0;   // Stage-2 int8 inference
    int64_t post_us = 0;    // dequantise + threshold
    int tiles_run = 0;
    int tiles_skipped = 0;
    int detections = 0;

    int64_t total_us() const
    {
        return fetch_us + gate_us + fill_us + infer_us + post_us;
    }
};

void log_memory(const char *label)
{
    ESP_LOGI(TAG,
             "%-22s internal free %7u (min %7u)  psram free %8u (min %8u)",
             label,
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
             static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM)));
}

// Quantise one tile into the model's int8 input buffer.
//
// ESP-DL runs Conv and the pooling operators in NHWC, so a 2-channel input is
// interleaved per pixel, not planar. For kInputChannels == 1 the two layouts
// coincide. Pixels are 0/1, so the float->int8 conversion collapses to a
// two-entry lookup table instead of per-pixel arithmetic.
void fill_input(int8_t *dst, const int8_t quant[2], const uint8_t *test_tile,
                const uint8_t *temp_tile, int stride, int tile)
{
    for (int y = 0; y < tile; ++y) {
        const uint8_t *test_row = test_tile + static_cast<size_t>(y) * stride;
        const uint8_t *temp_row = temp_tile + static_cast<size_t>(y) * stride;
        if (kInputChannels == 1) {
            int8_t *out = dst + static_cast<size_t>(y) * tile;
            for (int x = 0; x < tile; ++x) {
                out[x] = quant[test_row[x]];
            }
        } else {
            int8_t *out = dst + static_cast<size_t>(y) * tile * 2;
            for (int x = 0; x < tile; ++x) {
                out[2 * x + 0] = quant[test_row[x]];
                out[2 * x + 1] = quant[temp_row[x]];
            }
        }
    }
}

// One pass over a board's 25 tiles. `gate_enabled` is the only thing that
// differs between the `reference` and `gate_only` rows of the run matrix.
StageTiming run_board(dl::Model *model, const pcb::Board &board,
                      uint8_t *gate_scratch, bool gate_enabled, bool verbose)
{
    StageTiming timing;

    dl::TensorBase *input = model->get_inputs().begin()->second;
    dl::TensorBase *output = model->get_outputs().begin()->second;
    int8_t *input_data = input->get_element_ptr<int8_t>();

    const int input_exponent = static_cast<int>(input->exponent);
    const int8_t quant[2] = {
        dl::quantize<int8_t>(0.0f, DL_RESCALE(input_exponent)),
        dl::quantize<int8_t>(1.0f, DL_RESCALE(input_exponent)),
    };
    const float output_scale = DL_SCALE(static_cast<int>(output->exponent));

    for (int index = 0; index < pcb::kTilesPerBoard; ++index) {
        int64_t mark = esp_timer_get_time();
        const size_t offset = pcb::tile_offset(index);
        const uint8_t *test_tile = board.test + offset;
        const uint8_t *temp_tile = board.temp + offset;
        timing.fetch_us += esp_timer_get_time() - mark;

        if (gate_enabled) {
            mark = esp_timer_get_time();
            const int score = pcb::gate_score(test_tile, temp_tile, pcb::kBoardSize,
                                              pcb::kTileSize, pcb::kGateRadius,
                                              gate_scratch);
            timing.gate_us += esp_timer_get_time() - mark;
            if (!pcb::gate_fires(score, kGateThreshold)) {
                // A skipped tile produces no predictions at all. Any defect it
                // held is lost, and the reported recall must carry that loss.
                timing.tiles_skipped++;
                continue;
            }
        }

        mark = esp_timer_get_time();
        fill_input(input_data, quant, test_tile, temp_tile, pcb::kBoardSize,
                   pcb::kTileSize);
        timing.fill_us += esp_timer_get_time() - mark;

        mark = esp_timer_get_time();
        model->run();
        timing.infer_us += esp_timer_get_time() - mark;
        timing.tiles_run++;

        mark = esp_timer_get_time();
        const int8_t *logits = output->get_element_ptr<int8_t>();
        for (int c = 0; c < 6; ++c) {
            const float logit = dl::dequantize(logits[c], output_scale);
            if (logit > kLogitThresholds[c]) {
                timing.detections++;
                if (verbose) {
                    ESP_LOGI(TAG, "  tile %2d: %-10s logit %+.3f", index,
                             kClassNames[c], logit);
                }
            }
        }
        timing.post_us += esp_timer_get_time() - mark;
    }
    return timing;
}

void report(const char *label, const StageTiming &t, int boards)
{
    const int tiles = pcb::kTilesPerBoard * boards;
    const double per_board = static_cast<double>(t.total_us()) / boards / 1000.0;
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "--- %s (%d boards, %d tiles) ---", label, boards, tiles);
    ESP_LOGI(TAG, "  tiles run %d, skipped %d (%.1f%% of tiles skipped)",
             t.tiles_run, t.tiles_skipped,
             100.0 * t.tiles_skipped / tiles);
    ESP_LOGI(TAG, "  fetch  %8" PRId64 " us", t.fetch_us);
    ESP_LOGI(TAG, "  gate   %8" PRId64 " us  (%.1f us/tile)", t.gate_us,
             tiles ? static_cast<double>(t.gate_us) / tiles : 0.0);
    ESP_LOGI(TAG, "  fill   %8" PRId64 " us", t.fill_us);
    ESP_LOGI(TAG, "  infer  %8" PRId64 " us  (%.1f us per run tile)", t.infer_us,
             t.tiles_run ? static_cast<double>(t.infer_us) / t.tiles_run : 0.0);
    ESP_LOGI(TAG, "  post   %8" PRId64 " us", t.post_us);
    ESP_LOGI(TAG, "  TOTAL  %8" PRId64 " us  => %.2f ms per board",
             t.total_us(), per_board);
    ESP_LOGI(TAG, "  detections %d", t.detections);
}

}  // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "PCB defect cascade benchmark -- input channels %d, gate radius %d",
             kInputChannels, pcb::kGateRadius);
    log_memory("baseline");

    pcb::Board board = {};
    if (!pcb::alloc_board(&board)) {
        ESP_LOGE(TAG, "could not allocate a 640x640 board pair in PSRAM; "
                      "is CONFIG_SPIRAM enabled?");
        return;
    }
    pcb::make_synthetic_board(board, 12345u, 8);
    log_memory("after board alloc");

    auto *model = new dl::Model(reinterpret_cast<const char *>(model_espdl_start),
                               fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    log_memory("after model load");

    // Parity: device output vs the PC values ESP-PPQ embedded in the model.
    if (model->test() == ESP_OK) {
        ESP_LOGI(TAG, "dl::Model::test() PASSED -- device matches PC int8 output");
    } else {
        ESP_LOGE(TAG, "dl::Model::test() FAILED -- device and PC int8 disagree. "
                      "Do not trust any accuracy number until this passes.");
    }

    // Per-layer latency and the memory breakdown (fbs_model, parameter_copy,
    // variable, others) across internal RAM, PSRAM and flash.
    model->profile();

    auto *gate_scratch = static_cast<uint8_t *>(
        heap_caps_malloc(pcb::gate_scratch_bytes(), MALLOC_CAP_INTERNAL));
    if (gate_scratch == nullptr) {
        ESP_LOGE(TAG, "could not allocate %u bytes of gate scratch",
                 static_cast<unsigned>(pcb::gate_scratch_bytes()));
        return;
    }

    constexpr int kWarmupBoards = 1;
    constexpr int kBoards = 10;

    for (int i = 0; i < kWarmupBoards; ++i) {
        run_board(model, board, gate_scratch, false, false);
    }

    StageTiming gate_off;
    StageTiming gate_on;
    for (int i = 0; i < kBoards; ++i) {
        const StageTiming off = run_board(model, board, gate_scratch, false, false);
        gate_off.fetch_us += off.fetch_us;
        gate_off.gate_us += off.gate_us;
        gate_off.fill_us += off.fill_us;
        gate_off.infer_us += off.infer_us;
        gate_off.post_us += off.post_us;
        gate_off.tiles_run += off.tiles_run;
        gate_off.tiles_skipped += off.tiles_skipped;
        gate_off.detections += off.detections;

        const StageTiming on = run_board(model, board, gate_scratch, true, false);
        gate_on.fetch_us += on.fetch_us;
        gate_on.gate_us += on.gate_us;
        gate_on.fill_us += on.fill_us;
        gate_on.infer_us += on.infer_us;
        gate_on.post_us += on.post_us;
        gate_on.tiles_run += on.tiles_run;
        gate_on.tiles_skipped += on.tiles_skipped;
        gate_on.detections += on.detections;
    }

    report("gate OFF (all 25 tiles)", gate_off, kBoards);
    report("gate ON (eroded XOR)", gate_on, kBoards);

    const double off_ms = static_cast<double>(gate_off.total_us()) / kBoards / 1000.0;
    const double on_ms = static_cast<double>(gate_on.total_us()) / kBoards / 1000.0;
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "gate speedup %.2fx (%.2f ms -> %.2f ms per board)",
             on_ms > 0.0 ? off_ms / on_ms : 0.0, off_ms, on_ms);

    log_memory("final");
    ESP_LOGI(TAG, "PEAK internal SRAM used %u bytes, PSRAM used %u bytes",
             static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_INTERNAL)
                                   - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)
                                   - heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM)));

    heap_caps_free(gate_scratch);
    delete model;
    pcb::free_board(&board);
    ESP_LOGI(TAG, "benchmark complete");
}
