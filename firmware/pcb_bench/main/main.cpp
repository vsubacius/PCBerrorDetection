// Benchmark mode for the gated tile cascade on a Seeed XIAO ESP32-S3 Sense.
//
// Every number this project reports for the chip comes from here. The app:
//
//   1. records a memory baseline before anything is allocated;
//   2. loads the int8 .espdl student and runs dl::Model::test(), which compares
//      on-device output against the PC values ESP-PPQ embedded in the model --
//      the PC-vs-device parity check the experiment plan requires;
//   3. calls dl::Model::profile() for per-layer latency and a memory breakdown;
//   4. runs every embedded *real* DeepPCB validation board (models/boards.bin,
//      written by tools/make_boards.py) with the gate off and then on, timing
//      image access, gate, Stage-2 inference and postprocessing separately;
//   5. scores the chip's own predictions exactly as eval/metrics.py does
//      (defect-level recall, (tile, class) TP/FP/ignored, clean-board false
//      alarms) and counts how many (tile, class) decisions differ from the PC
//      int8 model;
//   6. reports peak (minimum-free) internal SRAM and PSRAM.
//
// The gate is a *runtime* switch, deliberately: the experiment plan requires
// `reference` and `gate_only` to differ in nothing but the gate, so both rows
// come from one firmware binary and one model file.
//
// Accuracy here is on 10 boards only. It is a smoke test that the deployed
// model behaves like the PC model, not a reported result.

#include <inttypes.h>
#include <stdio.h>

#include <algorithm>

#include "board.hpp"
#include "boards.hpp"
#include "camera.hpp"
#include "live.hpp"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "dl_model_base.hpp"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "gate.hpp"

namespace {

const char *TAG = "pcb_bench";

// Gate threshold on the eroded-XOR count. Selected on the validation fold in
// Python, never here. The gate fires when score > threshold.
constexpr int kGateThreshold = 5;

const char *kClassNames[pcb::kNumClasses] = {"open", "short", "mousebite",
                                             "spur", "copper", "pin-hole"};

// Both files are embedded in .rodata by main/CMakeLists.txt. The symbol is
// "_binary_" + filename with dots replaced by underscores + "_start"/"_end".
extern "C" const uint8_t model_espdl_start[] asm("_binary_student_c2_espdl_start");
extern "C" const uint8_t boards_bin_start[] asm("_binary_boards_bin_start");
extern "C" const uint8_t boards_bin_end[] asm("_binary_boards_bin_end");
extern "C" const uint8_t model_espdl_end[] asm("_binary_student_c2_espdl_end");

// --- machine-readable results: one "@@RESULT {json}" line per test --------
// tools/monitor.py appends each record to results/<kind>.csv and
// results/results.jsonl. Field names are the CSV column names.
struct Record {
    char buf[2048];
    int n = 0;
    bool first = true;

    explicit Record(const char *kind)
    {
        n = snprintf(buf, sizeof(buf), "{\"kind\":\"%s\",\"uptime_s\":%.1f", kind,
                     esp_timer_get_time() / 1e6);
    }
    void add(const char *key, double v)
    {
        if (v != v || v > 1e300 || v < -1e300) {   // NaN / inf are not valid JSON
            n += snprintf(buf + n, sizeof(buf) - n, ",\"%s\":null", key);
        } else {
            n += snprintf(buf + n, sizeof(buf) - n, ",\"%s\":%.6g", key, v);
        }
    }
    void add_int(const char *key, long long v)
    {
        n += snprintf(buf + n, sizeof(buf) - n, ",\"%s\":%lld", key, v);
    }
    void add_str(const char *key, const char *v)
    {
        n += snprintf(buf + n, sizeof(buf) - n, ",\"%s\":\"%s\"", key, v);
    }
    void add_memory()
    {
        add_int("sram_free_min_bytes", heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
        add_int("psram_free_min_bytes", heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
        add_int("sram_peak_used_bytes",
                heap_caps_get_total_size(MALLOC_CAP_INTERNAL)
                    - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
        add_int("psram_peak_used_bytes",
                heap_caps_get_total_size(MALLOC_CAP_SPIRAM)
                    - heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    }
    void emit()
    {
        if (n < static_cast<int>(sizeof(buf)) - 2) {
            buf[n++] = '}';
            buf[n] = 0;
        }
        printf("@@RESULT %s\n", buf);
        fflush(stdout);
    }
};

struct StageTiming {
    int64_t fetch_us = 0;   // reaching the tile's pixels
    int64_t gate_us = 0;    // Stage-1 eroded-XOR gate
    int64_t fill_us = 0;    // quantising the tile into the model input
    int64_t infer_us = 0;   // Stage-2 int8 inference
    int64_t post_us = 0;    // dequantise + threshold
    int tiles_run = 0;
    int tiles_skipped = 0;
    int boards = 0;

    int64_t total_us() const
    {
        return fetch_us + gate_us + fill_us + infer_us + post_us;
    }

    void add(const StageTiming &o)
    {
        fetch_us += o.fetch_us;
        gate_us += o.gate_us;
        fill_us += o.fill_us;
        infer_us += o.infer_us;
        post_us += o.post_us;
        tiles_run += o.tiles_run;
        tiles_skipped += o.tiles_skipped;
        boards += o.boards;
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
// interleaved per pixel, not planar. Pixels are 0/1, so the float->int8
// conversion collapses to a two-entry lookup table.
//   mode A: [test]   mode B: [test, template]   mode C: [test, test XOR template]
void fill_input(int8_t *dst, const int8_t quant[2], char rep,
                const uint8_t *test_tile, const uint8_t *temp_tile, int stride,
                int tile)
{
    for (int y = 0; y < tile; ++y) {
        const uint8_t *test_row = test_tile + static_cast<size_t>(y) * stride;
        const uint8_t *temp_row = temp_tile + static_cast<size_t>(y) * stride;
        if (rep == 'A') {
            int8_t *out = dst + static_cast<size_t>(y) * tile;
            for (int x = 0; x < tile; ++x) {
                out[x] = quant[test_row[x]];
            }
        } else if (rep == 'B') {
            int8_t *out = dst + static_cast<size_t>(y) * tile * 2;
            for (int x = 0; x < tile; ++x) {
                out[2 * x + 0] = quant[test_row[x]];
                out[2 * x + 1] = quant[temp_row[x]];
            }
        } else {
            int8_t *out = dst + static_cast<size_t>(y) * tile * 2;
            for (int x = 0; x < tile; ++x) {
                out[2 * x + 0] = quant[test_row[x]];
                out[2 * x + 1] = quant[test_row[x] ^ temp_row[x]];
            }
        }
    }
}

struct RunConfig {
    char rep;
    const float *thresholds;   // logit space, from the blob
};

// One pass over a board's 25 tiles. `gate_enabled` is the only thing that
// differs between the `reference` and `gate_only` rows of the run matrix.
StageTiming run_board(dl::Model *model, const RunConfig &cfg,
                      const pcb::Board &board, uint8_t *gate_scratch,
                      bool gate_enabled, pcb::BoardOutput *out,
                      float logits_out[pcb::kTilesPerBoard][pcb::kNumClasses])
{
    StageTiming timing;
    timing.boards = 1;

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
        out->fired[index] = 0;
        out->ran[index] = false;

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
                // held is lost, and the reported recall carries that loss.
                timing.tiles_skipped++;
                continue;
            }
        }

        mark = esp_timer_get_time();
        fill_input(input_data, quant, cfg.rep, test_tile, temp_tile,
                   pcb::kBoardSize, pcb::kTileSize);
        timing.fill_us += esp_timer_get_time() - mark;

        mark = esp_timer_get_time();
        model->run();
        timing.infer_us += esp_timer_get_time() - mark;
        timing.tiles_run++;

        mark = esp_timer_get_time();
        const int8_t *logits = output->get_element_ptr<int8_t>();
        uint8_t fired = 0;
        for (int c = 0; c < pcb::kNumClasses; ++c) {
            const float logit = dl::dequantize(logits[c], output_scale);
            if (logits_out != nullptr) {
                logits_out[index][c] = logit;
            }
            if (logit > cfg.thresholds[c]) {
                fired |= static_cast<uint8_t>(1u << c);
            }
        }
        out->fired[index] = fired;
        out->ran[index] = true;
        timing.post_us += esp_timer_get_time() - mark;
    }
    return timing;
}

// Per-detection detail for one board (gate off): what fired, and what it was.
// `ref` names what expect_mask came from and `self` names this run: in the
// benchmark that is the PC int8 model vs the chip; in live mode it is the
// stored image vs the camera image.
void log_board_detail(const pcb::BlobBoardMeta &meta, const pcb::BoardOutput &out,
                      const float logits[pcb::kTilesPerBoard][pcb::kNumClasses],
                      const char *ref = "PC int8", const char *self = "chip")
{
    for (int t = 0; t < pcb::kTilesPerBoard; ++t) {
        for (int c = 0; c < pcb::kNumClasses; ++c) {
            const uint8_t bit = static_cast<uint8_t>(1u << c);
            if (!(out.fired[t] & bit)) {
                continue;
            }
            const char *kind = (meta.y_mask[t] & bit)        ? "TP"
                               : (meta.ignore_mask[t] & bit) ? "ignored"
                                                             : "FP";
            const bool ref_fired = meta.expect_mask[t] & bit;
            ESP_LOGI(TAG, "    tile %2d  %-9s logit %+6.2f  %-7s%s%s%s", t,
                     kClassNames[c], logits[t][c], kind,
                     meta.has_expect && !ref_fired ? "  (" : "",
                     meta.has_expect && !ref_fired ? ref : "",
                     meta.has_expect && !ref_fired ? " did NOT fire)" : "");
        }
        if (meta.has_expect) {
            const uint8_t missing = meta.expect_mask[t] & static_cast<uint8_t>(~out.fired[t]);
            for (int c = 0; c < pcb::kNumClasses; ++c) {
                if (missing & (1u << c)) {
                    ESP_LOGW(TAG, "    tile %2d  %-9s logit %+6.2f  %s fired, %s did not",
                             t, kClassNames[c], logits[t][c], ref, self);
                }
            }
        }
    }
    for (uint32_t d = 0; d < meta.n_defects && d < pcb::kMaxDefects; ++d) {
        const pcb::BlobDefect &def = meta.defects[d];
        bool found = false;
        for (int t = 0; t < pcb::kTilesPerBoard; ++t) {
            if ((def.tile_mask & (1u << t)) && (out.fired[t] & (1u << def.cls))) {
                found = true;
            }
        }
        if (!found) {
            ESP_LOGI(TAG, "    MISSED  %-9s (qualifying tile mask 0x%07" PRIx32 ")",
                     kClassNames[def.cls], def.tile_mask);
        }
    }
}

void report_timing(const char *label, const StageTiming &t)
{
    const int tiles = pcb::kTilesPerBoard * t.boards;
    const double per_board = t.boards ? static_cast<double>(t.total_us()) / t.boards / 1000.0 : 0.0;
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "--- timing: %s (%d boards, %d tiles) ---", label, t.boards, tiles);
    ESP_LOGI(TAG, "  tiles run %d, skipped %d (%.1f%% of tiles skipped)",
             t.tiles_run, t.tiles_skipped, tiles ? 100.0 * t.tiles_skipped / tiles : 0.0);
    ESP_LOGI(TAG, "  fetch  %9" PRId64 " us", t.fetch_us);
    ESP_LOGI(TAG, "  gate   %9" PRId64 " us  (%.1f us/tile)", t.gate_us,
             tiles ? static_cast<double>(t.gate_us) / tiles : 0.0);
    ESP_LOGI(TAG, "  fill   %9" PRId64 " us", t.fill_us);
    ESP_LOGI(TAG, "  infer  %9" PRId64 " us  (%.2f ms per run tile)", t.infer_us,
             t.tiles_run ? static_cast<double>(t.infer_us) / t.tiles_run / 1000.0 : 0.0);
    ESP_LOGI(TAG, "  post   %9" PRId64 " us", t.post_us);
    ESP_LOGI(TAG, "  TOTAL  %9" PRId64 " us  => %.2f ms per board", t.total_us(), per_board);
}

void report_score(const char *label, const pcb::Score &s)
{
    const int fired = s.true_positives + s.false_positives;
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "--- accuracy: %s ---", label);
    ESP_LOGI(TAG, "  defects found %d / %d  (recall %.1f%%; %d lost to gate, %d unreachable)",
             s.n_found, s.n_defects,
             s.n_defects ? 100.0 * s.n_found / s.n_defects : 0.0,
             s.n_lost_to_gate, s.n_unreachable);
    for (int c = 0; c < pcb::kNumClasses; ++c) {
        ESP_LOGI(TAG, "    %-9s %d / %d", kClassNames[c], s.per_class_found[c],
                 s.per_class_defects[c]);
    }
    ESP_LOGI(TAG, "  (tile,class) firings: TP %d  FP %d  ignored %d  -> precision %.1f%%",
             s.true_positives, s.false_positives, s.ignored_firings,
             fired ? 100.0 * s.true_positives / fired : 0.0);
    ESP_LOGI(TAG, "  clean boards with any alarm: %d / %d",
             s.clean_boards_with_alarm, s.clean_boards);
    if (s.parity_checked) {
        ESP_LOGI(TAG, "  decisions differing from PC int8: %d / %d",
                 s.parity_mismatch, s.parity_checked);
    }
}

}  // namespace

// One full benchmark. Everything it allocates is freed before it returns, so
// app_main can run it again and again.
void run_benchmark()
{
    ESP_LOGI(TAG, "PCB defect cascade benchmark -- real DeepPCB boards, gate radius %d, "
                  "gate threshold %d", pcb::kGateRadius, kGateThreshold);
    log_memory("baseline");

    pcb::BoardSet set;
    const size_t blob_size = static_cast<size_t>(boards_bin_end - boards_bin_start);
    if (!pcb::parse_board_blob(boards_bin_start, blob_size, &set)) {
        ESP_LOGE(TAG, "models/boards.bin is missing or malformed (%u bytes); "
                      "rerun tools/make_boards.py", static_cast<unsigned>(blob_size));
        return;
    }
    const RunConfig cfg = {static_cast<char>(set.header->representation),
                           set.header->thresholds_logit};
    ESP_LOGI(TAG, "%u boards, input mode %c (%u channels), thresholds (logit) "
                  "%.3f %.3f %.3f %.3f %.3f %.3f",
             static_cast<unsigned>(set.header->n_boards), cfg.rep,
             static_cast<unsigned>(set.header->in_channels),
             cfg.thresholds[0], cfg.thresholds[1], cfg.thresholds[2],
             cfg.thresholds[3], cfg.thresholds[4], cfg.thresholds[5]);

    pcb::Board board = {};
    if (!pcb::alloc_board(&board)) {
        ESP_LOGE(TAG, "could not allocate a 640x640 board pair in PSRAM; "
                      "is CONFIG_SPIRAM enabled?");
        return;
    }
    log_memory("after board alloc");

    auto *model = new dl::Model(reinterpret_cast<const char *>(model_espdl_start),
                               fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    log_memory("after model load");

    // The model's input must match the blob's input mode.
    dl::TensorBase *input = model->get_inputs().begin()->second;
    const int model_channels = input->shape.back();   // NHWC
    if (model_channels != static_cast<int>(set.header->in_channels)) {
        ESP_LOGE(TAG, "model expects %d input channels but boards.bin was made for %u",
                 model_channels, static_cast<unsigned>(set.header->in_channels));
        return;
    }

    // Parity: device output vs the PC values ESP-PPQ embedded in the model.
    if (model->test() == ESP_OK) {
        ESP_LOGI(TAG, "dl::Model::test() PASSED -- device matches PC int8 output");
    } else {
        ESP_LOGE(TAG, "dl::Model::test() FAILED -- device and PC int8 disagree. "
                      "Do not trust any accuracy number until this passes.");
    }

    // Per-layer latency and the memory breakdown.
    model->profile();

    auto *gate_scratch = static_cast<uint8_t *>(
        heap_caps_malloc(pcb::gate_scratch_bytes(), MALLOC_CAP_INTERNAL));
    if (gate_scratch == nullptr) {
        ESP_LOGE(TAG, "could not allocate %u bytes of gate scratch",
                 static_cast<unsigned>(pcb::gate_scratch_bytes()));
        return;
    }

    static float logits[pcb::kTilesPerBoard][pcb::kNumClasses];
    pcb::BoardOutput out;

    // Warm-up pass (caches, lazy allocations) -- not timed.
    pcb::unpack_board(set, 0, board);
    run_board(model, cfg, board, gate_scratch, false, &out, nullptr);

    StageTiming time_off, time_on;
    pcb::Score score_off, score_on;

    for (uint32_t b = 0; b < set.header->n_boards; ++b) {
        const pcb::BlobBoardMeta &meta = set.meta[b];
        pcb::unpack_board(set, static_cast<int>(b), board);

        const StageTiming t_off = run_board(model, cfg, board, gate_scratch, false,
                                            &out, logits);
        const pcb::Score s_off = pcb::score_board(meta, out);
        ESP_LOGI(TAG, "");
        ESP_LOGI(TAG, "board %-10s %-9s gate OFF: found %d/%d defects, TP %d FP %d "
                      "ign %d, PC mismatches %d/%d, %.1f ms",
                 meta.name, meta.is_clean ? "(clean)" : "", s_off.n_found,
                 s_off.n_defects, s_off.true_positives, s_off.false_positives,
                 s_off.ignored_firings, s_off.parity_mismatch, s_off.parity_checked,
                 t_off.total_us() / 1000.0);
        log_board_detail(meta, out, logits);

        const StageTiming t_on = run_board(model, cfg, board, gate_scratch, true,
                                           &out, nullptr);
        const pcb::Score s_on = pcb::score_board(meta, out);
        ESP_LOGI(TAG, "  gate ON: ran %d/25 tiles, found %d/%d defects (%d lost to gate), "
                      "FP %d, %.1f ms",
                 t_on.tiles_run, s_on.n_found, s_on.n_defects, s_on.n_lost_to_gate,
                 s_on.false_positives, t_on.total_us() / 1000.0);
        {
            Record rec("bench_board");
            rec.add_str("board", meta.name);
            rec.add_int("clean", meta.is_clean);
            rec.add_int("defects", s_off.n_defects);
            rec.add_int("off_found", s_off.n_found);
            rec.add_int("off_tp", s_off.true_positives);
            rec.add_int("off_fp", s_off.false_positives);
            rec.add_int("off_ignored", s_off.ignored_firings);
            rec.add_int("off_decisions_differ_from_pc_int8", s_off.parity_mismatch);
            rec.add("off_ms", t_off.total_us() / 1000.0);
            rec.add_int("on_tiles_run", t_on.tiles_run);
            rec.add_int("on_found", s_on.n_found);
            rec.add_int("on_lost_to_gate", s_on.n_lost_to_gate);
            rec.add_int("on_fp", s_on.false_positives);
            rec.add("on_ms", t_on.total_us() / 1000.0);
            rec.add("on_gate_ms", t_on.gate_us / 1000.0);
            rec.emit();
        }

        // Let the idle task run between boards (outside every timed region),
        // so the task watchdog does not fire during the long benchmark.
        vTaskDelay(1);

        time_off.add(t_off);
        time_on.add(t_on);
        score_off.add(s_off);
        score_on.add(s_on);
    }

    report_score("gate OFF (all 25 tiles)", score_off);
    report_score("gate ON (eroded XOR)", score_on);
    report_timing("gate OFF (all 25 tiles)", time_off);
    report_timing("gate ON (eroded XOR)", time_on);

    const double off_ms = time_off.boards ? time_off.total_us() / 1000.0 / time_off.boards : 0.0;
    const double on_ms = time_on.boards ? time_on.total_us() / 1000.0 / time_on.boards : 0.0;
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "gate speedup %.2fx (%.2f ms -> %.2f ms per board)",
             on_ms > 0.0 ? off_ms / on_ms : 0.0, off_ms, on_ms);

    log_memory("final");
    ESP_LOGI(TAG, "PEAK internal SRAM used %u bytes, PSRAM used %u bytes",
             static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_INTERNAL)
                                   - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)
                                   - heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM)));
    {
        auto add_timing = [](Record &rec, const char *p, const StageTiming &t) {
            char k[48];
            const int tiles = pcb::kTilesPerBoard * t.boards;
            snprintf(k, sizeof(k), "%s_ms_per_board", p);
            rec.add(k, t.boards ? t.total_us() / 1000.0 / t.boards : 0.0);
            snprintf(k, sizeof(k), "%s_infer_ms_per_tile", p);
            rec.add(k, t.tiles_run ? t.infer_us / 1000.0 / t.tiles_run : 0.0);
            snprintf(k, sizeof(k), "%s_fill_ms_per_tile", p);
            rec.add(k, t.tiles_run ? t.fill_us / 1000.0 / t.tiles_run : 0.0);
            snprintf(k, sizeof(k), "%s_gate_ms_per_tile", p);
            rec.add(k, tiles ? t.gate_us / 1000.0 / tiles : 0.0);
            snprintf(k, sizeof(k), "%s_tiles_run", p);
            rec.add_int(k, t.tiles_run);
            snprintf(k, sizeof(k), "%s_tiles_skipped", p);
            rec.add_int(k, t.tiles_skipped);
        };
        auto add_score = [](Record &rec, const char *p, const pcb::Score &sc) {
            char k[48];
            snprintf(k, sizeof(k), "%s_recall", p);
            rec.add(k, sc.n_defects ? static_cast<double>(sc.n_found) / sc.n_defects : 0.0);
            snprintf(k, sizeof(k), "%s_found", p);
            rec.add_int(k, sc.n_found);
            snprintf(k, sizeof(k), "%s_lost_to_gate", p);
            rec.add_int(k, sc.n_lost_to_gate);
            snprintf(k, sizeof(k), "%s_tp", p);
            rec.add_int(k, sc.true_positives);
            snprintf(k, sizeof(k), "%s_fp", p);
            rec.add_int(k, sc.false_positives);
            snprintf(k, sizeof(k), "%s_precision", p);
            const int fired = sc.true_positives + sc.false_positives;
            rec.add(k, fired ? static_cast<double>(sc.true_positives) / fired : 0.0);
            snprintf(k, sizeof(k), "%s_clean_boards_with_alarm", p);
            rec.add_int(k, sc.clean_boards_with_alarm);
            snprintf(k, sizeof(k), "%s_decisions_differ_from_pc_int8", p);
            rec.add_int(k, sc.parity_mismatch);
            snprintf(k, sizeof(k), "%s_decisions_checked", p);
            rec.add_int(k, sc.parity_checked);
        };
        Record rec("bench_summary");
        rec.add_int("boards", time_off.boards);
        rec.add_int("defects", score_off.n_defects);
        rec.add_int("clean_boards", score_off.clean_boards);
        add_score(rec, "off", score_off);
        add_score(rec, "on", score_on);
        add_timing(rec, "off", time_off);
        add_timing(rec, "on", time_on);
        rec.add("gate_speedup", on_ms > 0.0 ? off_ms / on_ms : 0.0);
        rec.add_int("model_flash_bytes", model_espdl_end - model_espdl_start);
        rec.add_int("cpu_mhz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
        rec.add_memory();
        rec.emit();
    }

    heap_caps_free(gate_scratch);
    delete model;
    pcb::free_board(&board);
    ESP_LOGI(TAG, "benchmark complete");
}

// ================================================================ live mode
//
// Commands arrive as single characters over the USB serial port (type the
// letter and press Enter in tools/monitor.py). The camera looks at a printed
// sheet from tools/make_prints.py; the operator types the board number shown
// on the sheet, then 'c' to capture. The stored copy of the same board gives
// the template, the labels, and a reference result to compare with.

namespace {

struct LiveState {
    pcb::BoardSet set;
    RunConfig cfg = {};
    dl::Model *model = nullptr;
    pcb::Board board = {};
    uint8_t *gate_scratch = nullptr;
    uint8_t *frame = nullptr;        // last captured frame (largest size)
    int frame_w = 0;
    int frame_h = 0;
    uint8_t *warp = nullptr;         // live::kWarpSize^2
    uint16_t *flat_cells = nullptr;  // recorded blank-paper flat field
    pcb::live::FlatField flat;
    bool flat_valid = false;
    int selected = 0;
    bool camera_ok = false;
};

LiveState g;

// --- image dumps: "@@IMG name w h gray|bit nbytes", base64 lines, "@@END crc32" ---
//
// The camera driver prints warnings ("cam_hal: ...") straight to the console
// from its own task, so they can land inside the base64 lines. tools/monitor.py
// removes them, and checks the byte count and the CRC32 (same as zlib.crc32).

uint32_t crc32(const uint8_t *data, size_t n)
{
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
        ready = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

void base64_stream(const uint8_t *data, size_t n)
{
    static const char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char line[80];
    int pos = 0;
    int lines = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < n ? data[i + 1] : 0;
        const uint32_t b2 = i + 2 < n ? data[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        line[pos++] = kTable[(v >> 18) & 63];
        line[pos++] = kTable[(v >> 12) & 63];
        line[pos++] = i + 1 < n ? kTable[(v >> 6) & 63] : '=';
        line[pos++] = i + 2 < n ? kTable[v & 63] : '=';
        if (pos >= 76) {
            line[pos++] = '\n';
            fwrite(line, 1, pos, stdout);
            pos = 0;
            // A full-size frame takes several seconds to send. Let the idle
            // task run now and then, or the task watchdog fires and prints its
            // report straight into the middle of the image data.
            if (++lines % 128 == 0) {
                fflush(stdout);
                vTaskDelay(1);
            }
        }
    }
    if (pos > 0) {
        line[pos++] = '\n';
        fwrite(line, 1, pos, stdout);
    }
}

void dump_gray(const char *name, const uint8_t *gray, int w, int h)
{
    printf("@@IMG %s %d %d gray %d\n", name, w, h, w * h);
    base64_stream(gray, static_cast<size_t>(w) * h);
    printf("@@END %08" PRIx32 "\n", crc32(gray, static_cast<size_t>(w) * h));
    fflush(stdout);
}

// 640x640 0/1 bytes -> packed bits (MSB first) in chunks.
void dump_board(const char *name, const uint8_t *board01)
{
    constexpr int kBytes = pcb::kBoardSize * pcb::kBoardSize / 8;
    static uint8_t packed[kBytes];
    for (int i = 0; i < kBytes; ++i) {
        const uint8_t *p = board01 + static_cast<size_t>(i) * 8;
        packed[i] = static_cast<uint8_t>((p[0] << 7) | (p[1] << 6) | (p[2] << 5) | (p[3] << 4) |
                                         (p[4] << 3) | (p[5] << 2) | (p[6] << 1) | p[7]);
    }
    printf("@@IMG %s %d %d bit %d\n", name, pcb::kBoardSize, pcb::kBoardSize, kBytes);
    base64_stream(packed, kBytes);
    printf("@@END %08" PRIx32 "\n", crc32(packed, kBytes));
    fflush(stdout);
}

// --- commands ---

void print_help()
{
    printf("\n");
    printf("=============================== PCB live test ===============================\n");
    printf(" camera: %s, %s%s\n", g.camera_ok ? pcb::camera::sensor_name() : "NOT FOUND",
           g.camera_ok ? pcb::camera::size(pcb::camera::current_size()).name : "-",
           g.flat_valid ? ", flat field recorded" : ", automatic light correction");
    printf(" boards on the printed sheets (type the number):\n");
    for (uint32_t i = 0; i < g.set.header->n_boards; ++i) {
        const pcb::BlobBoardMeta &m = g.set.meta[i];
        printf("   %u  %-10s %s%s\n", static_cast<unsigned>(i), m.name,
               m.is_clean ? "clean (template printed)" : "defective",
               static_cast<int>(i) == g.selected ? "   <-- selected" : "");
    }
    printf(" commands (type, then Enter):\n");
    printf("   0-9  select the board that is under the camera\n");
    printf("   c    capture and test the selected board\n");
    printf("   v    live view on the PC screen; in the view: c = capture, 0-9 = board, Enter = stop\n");
    printf("   p    save a full-quality preview frame\n");
    printf("   w    record a blank white sheet as the flat field;  x  forget it\n");
    printf("   r    next camera resolution\n");
    printf("   b    run the stored-image benchmark (no camera)\n");
    printf("   e    energy test: 30 s running, 30 s idle (read a USB power meter)\n");
    printf("   h    this help\n");
    printf("==============================================================================\n");
    fflush(stdout);
}

double g_last_capture_ms = 0;

bool capture_frame()
{
    const int64_t t0 = esp_timer_get_time();
    if (!pcb::camera::capture(g.frame, &g.frame_w, &g.frame_h)) {
        return false;
    }
    g_last_capture_ms = (esp_timer_get_time() - t0) / 1000.0;
    ESP_LOGI(TAG, "captured %dx%d in %.0f ms", g.frame_w, g.frame_h, g_last_capture_ms);
    return true;
}

void normalize_frame()
{
    if (g.flat_valid && g.flat.w == g.frame_w && g.flat.h == g.frame_h) {
        pcb::live::apply_flat_field(g.frame, g.frame_w, g.frame_h, g.flat);
        return;
    }
    // Automatic: paper brightness from the frame itself.
    static uint16_t cells[(1600 / 32 + 1) * (1200 / 32 + 1)];
    pcb::live::FlatField ff;
    ff.cell = 32;
    ff.mean = cells;
    const pcb::live::Frame f = {g.frame, g.frame_w, g.frame_h};
    pcb::live::build_auto_background(f, 3, &ff);
    pcb::live::apply_flat_field(g.frame, g.frame_w, g.frame_h, ff);
}

void command_flat_field()
{
    if (!g.camera_ok || !capture_frame()) {
        return;
    }
    g.flat.cell = 16;
    g.flat.mean = g.flat_cells;
    const pcb::live::Frame f = {g.frame, g.frame_w, g.frame_h};
    pcb::live::build_flat_field(f, &g.flat);
    uint16_t lo = 65535, hi = 0;
    for (int i = 0; i < g.flat.cw * g.flat.ch; ++i) {
        lo = std::min(lo, g.flat.mean[i]);
        hi = std::max(hi, g.flat.mean[i]);
    }
    g.flat_valid = true;
    ESP_LOGI(TAG, "flat field recorded at %dx%d: paper brightness %u (darkest) .. %u (brightest)",
             g.frame_w, g.frame_h, lo, hi);
    if (hi < 80) {
        ESP_LOGW(TAG, "the blank sheet is very dark -- add light, or the camera is covered");
    }
}

void command_preview()
{
    if (!g.camera_ok || !capture_frame()) {
        return;
    }
    dump_gray("preview", g.frame, g.frame_w, g.frame_h);
}

// --- live view: JPEG frames as "@@IMG live w h jpeg nbytes" until a key ---

int64_t g_view_start = 0;

// What the last key in the view asked for.
enum class ViewKey { kNone, kCapture, kStop };
ViewKey g_view_key = ViewKey::kNone;
uint8_t g_view_last = '\n';

bool view_should_stop()
{
    uint8_t ch;
    while (usb_serial_jtag_read_bytes(&ch, 1, 0) > 0) {
        if (ch == '\r') {
            continue;
        }
        const uint8_t last = g_view_last;
        g_view_last = ch;
        if (ch == '\n') {
            if (last == '\n') {   // an empty line: Enter alone stops the view
                g_view_key = ViewKey::kStop;
                return true;
            }
            continue;
        }
        if (ch == 'c') {
            g_view_key = ViewKey::kCapture;
            return true;
        }
        if (ch == 'q') {
            g_view_key = ViewKey::kStop;
            return true;
        }
        if (ch >= '0' && ch <= '9' && ch - '0' < static_cast<int>(g.set.header->n_boards)) {
            g.selected = ch - '0';
            ESP_LOGI(TAG, "selected board %d (%s)", g.selected, g.set.meta[g.selected].name);
        }
    }
    if (esp_timer_get_time() - g_view_start > 180LL * 1000 * 1000) {   // 3 min limit
        g_view_key = ViewKey::kStop;
        return true;
    }
    return false;
}

bool view_send_frame(const uint8_t *jpeg, size_t n, int w, int h)
{
    printf("@@IMG live %d %d jpeg %u board=%d\n", w, h, static_cast<unsigned>(n), g.selected);
    base64_stream(jpeg, n);
    printf("@@END %08" PRIx32 "\n", crc32(jpeg, n));
    fflush(stdout);
    vTaskDelay(1);   // idle task / watchdog
    return true;
}

void command_capture();

void drain_input()
{
    uint8_t junk;
    while (usb_serial_jtag_read_bytes(&junk, 1, pdMS_TO_TICKS(50)) > 0) {
    }
}

void command_view()
{
    if (!g.camera_ok) {
        ESP_LOGE(TAG, "no camera");
        return;
    }
    ESP_LOGI(TAG, "live view at %s. In the view: c = capture and test, 0-9 = pick the "
                  "board, Enter or q = stop (stops by itself after 3 min)",
             pcb::camera::size(pcb::camera::current_size()).name);
    for (;;) {
        // Drop the Enter typed after the last command, or it would count as a key.
        drain_input();
        g_view_last = '\n';
        g_view_key = ViewKey::kNone;
        g_view_start = esp_timer_get_time();
        const int frames = pcb::camera::stream_jpeg(view_send_frame, view_should_stop);
        const double secs = (esp_timer_get_time() - g_view_start) / 1e6;
        ESP_LOGI(TAG, "live view: %d frames in %.0f s (%.1f per second)", frames, secs,
                 secs > 0 ? frames / secs : 0.0);
        if (g_view_key != ViewKey::kCapture) {
            break;
        }
        command_capture();
        ESP_LOGI(TAG, "back to the live view (c = capture again, Enter = stop)");
    }
    ESP_LOGI(TAG, "live view stopped");
}

void command_capture()
{
    if (!g.camera_ok) {
        ESP_LOGE(TAG, "no camera");
        return;
    }
    const int index = g.selected;
    const pcb::BlobBoardMeta &meta = g.set.meta[index];
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "######## live test: board %d (%s, %s) ########", index, meta.name,
             meta.is_clean ? "clean" : "defective");

    if (!capture_frame()) {
        return;
    }
    vTaskDelay(1);
    int64_t t = esp_timer_get_time();
    normalize_frame();
    const double normalize_ms = (esp_timer_get_time() - t) / 1000.0;

    // Reference: the stored copy of the same board, gate off.
    pcb::unpack_board(g.set, index, g.board);
    pcb::BoardOutput ref_out;
    run_board(g.model, g.cfg, g.board, g.gate_scratch, false, &ref_out, nullptr);
    const pcb::Score ref_score = pcb::score_board(meta, ref_out);
    vTaskDelay(1);   // let the idle task run between long steps (task watchdog)

    // Align the frame to the template; board.test becomes the live image.
    t = esp_timer_get_time();
    const pcb::live::Frame f = {g.frame, g.frame_w, g.frame_h};
    pcb::live::AlignResult a = pcb::live::align_frame(f, g.board.temp, g.board.test, g.warp);
    const double align_ms = (esp_timer_get_time() - t) / 1000.0;
    vTaskDelay(1);
    ESP_LOGI(TAG, "light correction %.0f ms (%s), alignment %.0f ms", normalize_ms,
             g.flat_valid ? "recorded flat field" : "automatic", align_ms);
    ESP_LOGI(TAG, "frame threshold %u, %d dark blobs, %d square-ish", a.frame_threshold,
             a.n_blobs, a.n_candidates);
    if (!a.ok) {
        ESP_LOGE(TAG, "ALIGNMENT FAILED: %s", a.message);
        ESP_LOGE(TAG, "type p to save a preview frame and check the view");
        return;
    }
    ESP_LOGI(TAG, "corner squares at (%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f); "
                  "board is %.0f camera px wide (%.2f camera px per board px)",
             a.fiducial[0].x, a.fiducial[0].y, a.fiducial[1].x, a.fiducial[1].y,
             a.fiducial[2].x, a.fiducial[2].y, a.fiducial[3].x, a.fiducial[3].y,
             a.board_side_px, a.board_side_px / pcb::live::kBoardPx);
    ESP_LOGI(TAG, "orientation %d%s: %d of %d coarse samples differ from template "
                  "(next best orientation %d)",
             a.orientation & 3, a.orientation >= 4 ? " mirrored" : "", a.coarse_mismatch,
             a.coarse_samples, a.coarse_second);
    ESP_LOGI(TAG, "board threshold %u, sharpness %.1f, shift (%d,%d), %.2f%% of pixels "
                  "differ from template after shift",
             a.board_threshold, a.sharpness, a.shift_x, a.shift_y,
             100.0 * a.shift_mismatch / a.shift_samples);
    if (a.board_side_px < 0.45f * pcb::live::kBoardPx) {
        ESP_LOGW(TAG, "board is small in the frame (%.0f px): use a higher resolution (r)",
                 a.board_side_px);
    }

    // Camera error: live image vs the stored test image of the same board.
    const uint8_t *packed_test = g.set.images + static_cast<size_t>(index) * 2 * pcb::kPackedImageBytes;
    int pixel_err = 0;
    for (int i = 0; i < pcb::kBoardSize * pcb::kBoardSize; ++i) {
        const uint8_t bit = (packed_test[i >> 3] >> (7 - (i & 7))) & 1u;
        pixel_err += bit != g.board.test[i];
    }

    // Live passes.
    static float logits[pcb::kTilesPerBoard][pcb::kNumClasses];
    pcb::BoardOutput out_off, out_on;
    const StageTiming t_off = run_board(g.model, g.cfg, g.board, g.gate_scratch, false,
                                        &out_off, logits);
    const pcb::Score s_off = pcb::score_board(meta, out_off);
    vTaskDelay(1);
    const StageTiming t_on = run_board(g.model, g.cfg, g.board, g.gate_scratch, true,
                                       &out_on, nullptr);
    const pcb::Score s_on = pcb::score_board(meta, out_on);
    int changed = 0;
    for (int tile = 0; tile < pcb::kTilesPerBoard; ++tile) {
        changed += __builtin_popcount(out_off.fired[tile] ^ ref_out.fired[tile]);
    }

    ESP_LOGI(TAG, "---- result ----");
    log_board_detail(meta, out_off, logits, "stored image", "live");
    ESP_LOGI(TAG, "pixels different from the stored image: %d (%.2f%%)", pixel_err,
             100.0 * pixel_err / (pcb::kBoardSize * pcb::kBoardSize));
    ESP_LOGI(TAG, "LIVE   gate OFF: found %d/%d defects, TP %d FP %d ign %d, %.0f ms",
             s_off.n_found, s_off.n_defects, s_off.true_positives, s_off.false_positives,
             s_off.ignored_firings, t_off.total_us() / 1000.0);
    ESP_LOGI(TAG, "LIVE   gate ON : ran %d/25 tiles, found %d/%d (%d lost to gate), FP %d, %.0f ms",
             t_on.tiles_run, s_on.n_found, s_on.n_defects, s_on.n_lost_to_gate,
             s_on.false_positives, t_on.total_us() / 1000.0);
    ESP_LOGI(TAG, "STORED gate OFF: found %d/%d defects, TP %d FP %d ign %d",
             ref_score.n_found, ref_score.n_defects, ref_score.true_positives,
             ref_score.false_positives, ref_score.ignored_firings);
    ESP_LOGI(TAG, "(tile,class) decisions that changed, live vs stored: %d of 150", changed);

    char name[32];
    snprintf(name, sizeof(name), "board%d_live", index);
    dump_board(name, g.board.test);
    pcb::unpack_board(g.set, index, g.board);
    snprintf(name, sizeof(name), "board%d_stored", index);
    dump_board(name, g.board.test);

    Record rec("live_capture");
    rec.add_int("board_index", index);
    rec.add_str("board", meta.name);
    rec.add_int("clean", meta.is_clean);
    rec.add_str("camera", pcb::camera::sensor_name());
    rec.add_int("frame_w", g.frame_w);
    rec.add_int("frame_h", g.frame_h);
    rec.add_str("light", g.flat_valid ? "flat_field" : "auto");
    rec.add("board_camera_px", a.board_side_px);
    rec.add("camera_px_per_board_px", a.board_side_px / pcb::live::kBoardPx);
    rec.add("sharpness", a.sharpness);
    rec.add_int("orientation", a.orientation);
    rec.add("coarse_mismatch_pct", 100.0 * a.coarse_mismatch / a.coarse_samples);
    rec.add_int("shift_x", a.shift_x);
    rec.add_int("shift_y", a.shift_y);
    rec.add("pixel_error_pct", 100.0 * pixel_err / (pcb::kBoardSize * pcb::kBoardSize));
    rec.add_int("defects", s_off.n_defects);
    rec.add_int("live_found", s_off.n_found);
    rec.add_int("live_tp", s_off.true_positives);
    rec.add_int("live_fp", s_off.false_positives);
    rec.add_int("live_ignored", s_off.ignored_firings);
    rec.add_int("stored_found", ref_score.n_found);
    rec.add_int("stored_tp", ref_score.true_positives);
    rec.add_int("stored_fp", ref_score.false_positives);
    rec.add_int("stored_ignored", ref_score.ignored_firings);
    rec.add_int("decisions_changed", changed);
    rec.add_int("gate_on_tiles_run", t_on.tiles_run);
    rec.add_int("gate_on_found", s_on.n_found);
    rec.add_int("gate_on_lost_to_gate", s_on.n_lost_to_gate);
    rec.add_int("gate_on_fp", s_on.false_positives);
    rec.add("capture_ms", g_last_capture_ms);
    rec.add("light_ms", normalize_ms);
    rec.add("align_ms", align_ms);
    rec.add("gate_off_ms", t_off.total_us() / 1000.0);
    rec.add("gate_off_infer_ms_per_tile",
            t_off.tiles_run ? t_off.infer_us / 1000.0 / t_off.tiles_run : 0.0);
    rec.add("gate_on_ms", t_on.total_us() / 1000.0);
    rec.add("gate_on_gate_ms", t_on.gate_us / 1000.0);
    rec.add_memory();
    rec.emit();
}

// Energy: the model runs nonstop for 30 s, then the board idles for 30 s.
// Read a USB power meter in each phase; energy per tile = (P_run - P_idle) /
// tiles per second (or P_run / tiles per second for the whole board).
void command_energy()
{
    ESP_LOGI(TAG, "energy test: 30 s of nonstop inference (gate off, stored board 0), "
                  "then 30 s idle. Read your USB power meter in each phase.");
    pcb::unpack_board(g.set, 0, g.board);
    pcb::BoardOutput out;
    int tiles = 0;
    const int64_t t0 = esp_timer_get_time();
    int64_t next_note = t0;
    while (esp_timer_get_time() - t0 < 30LL * 1000 * 1000) {
        tiles += run_board(g.model, g.cfg, g.board, g.gate_scratch, false, &out, nullptr).tiles_run;
        if (esp_timer_get_time() >= next_note) {
            ESP_LOGI(TAG, "  RUNNING -- read the meter now (%d s left)",
                     static_cast<int>(30 - (esp_timer_get_time() - t0) / 1000000));
            next_note += 10LL * 1000 * 1000;
        }
        vTaskDelay(1);
    }
    const double run_s = (esp_timer_get_time() - t0) / 1e6;
    for (int left = 30; left > 0; left -= 10) {
        ESP_LOGI(TAG, "  IDLE -- read the meter now (%d s left)", left);
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
    const double tps = tiles / run_s;
    ESP_LOGI(TAG, "energy test done: %d tiles in %.1f s = %.1f tiles/s (%.2f ms per tile). "
                  "Energy per tile = power while running / %.1f", tiles, run_s, tps,
             1000.0 / tps, tps);
    Record rec("energy_run");
    rec.add_int("tiles", tiles);
    rec.add("run_s", run_s);
    rec.add("tiles_per_s", tps);
    rec.add("ms_per_tile", 1000.0 / tps);
    rec.add_int("cpu_mhz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    rec.emit();
}

void command_resolution()
{
    if (!g.camera_ok) {
        return;
    }
    const int next = (pcb::camera::current_size() + 1) % pcb::camera::num_sizes();
    if (pcb::camera::set_size(next)) {
        ESP_LOGI(TAG, "camera resolution now %s%s", pcb::camera::size(next).name,
                 g.flat_valid ? " (flat field cleared: record it again with w)" : "");
        g.flat_valid = false;
    }
}

bool live_setup()
{
    const size_t blob_size = static_cast<size_t>(boards_bin_end - boards_bin_start);
    if (!pcb::parse_board_blob(boards_bin_start, blob_size, &g.set)) {
        ESP_LOGE(TAG, "models/boards.bin is missing or malformed");
        return false;
    }
    g.cfg = {static_cast<char>(g.set.header->representation), g.set.header->thresholds_logit};
    if (!pcb::alloc_board(&g.board)) {
        ESP_LOGE(TAG, "board allocation failed");
        return false;
    }
    g.gate_scratch = static_cast<uint8_t *>(
        heap_caps_malloc(pcb::gate_scratch_bytes(), MALLOC_CAP_INTERNAL));
    g.frame = static_cast<uint8_t *>(heap_caps_malloc(1600 * 1200, MALLOC_CAP_SPIRAM));
    g.warp = static_cast<uint8_t *>(heap_caps_malloc(
        static_cast<size_t>(pcb::live::kWarpSize) * pcb::live::kWarpSize, MALLOC_CAP_SPIRAM));
    g.flat_cells = static_cast<uint16_t *>(heap_caps_malloc(
        sizeof(uint16_t) * (1600 / 16) * (1200 / 16), MALLOC_CAP_SPIRAM));
    if (!g.gate_scratch || !g.frame || !g.warp || !g.flat_cells) {
        ESP_LOGE(TAG, "buffer allocation failed");
        return false;
    }
    g.model = new dl::Model(reinterpret_cast<const char *>(model_espdl_start),
                            fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    // Long live-mode steps call this so the idle task (watchdog) can run.
    pcb::live::set_yield([]() { vTaskDelay(1); });
    g.camera_ok = pcb::camera::init();
    if (g.camera_ok) {
        ESP_LOGI(TAG, "camera: %s", pcb::camera::sensor_name());
    }
    Record rec("session");
    rec.add_str("camera", g.camera_ok ? pcb::camera::sensor_name() : "none");
    rec.add_int("model_flash_bytes", model_espdl_end - model_espdl_start);
    rec.add_str("input_mode", g.cfg.rep == 'C' ? "C" : g.cfg.rep == 'B' ? "B" : "A");
    rec.add_int("boards", g.set.header->n_boards);
    rec.add_int("cpu_mhz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    rec.add_int("gate_threshold", kGateThreshold);
    rec.add_memory();
    rec.emit();
    return true;
}

void setup_serial_input()
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 256;
    cfg.tx_buffer_size = 4096;
    if (usb_serial_jtag_driver_install(&cfg) == ESP_OK) {
        usb_serial_jtag_vfs_use_driver();
    }
}

}  // namespace

extern "C" void app_main(void)
{
    setup_serial_input();
    // The XIAO's USB port drops off the bus at every reset; give the PC
    // monitor time to reconnect before printing anything important.
    for (int i = 3; i > 0; --i) {
        ESP_LOGI(TAG, "starting in %d s", i);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    if (!live_setup()) {
        ESP_LOGE(TAG, "setup failed; running the stored-image benchmark only");
        run_benchmark();
        return;
    }
    print_help();

    int64_t last_input = esp_timer_get_time();
    for (;;) {
        uint8_t ch = 0;
        const int n = usb_serial_jtag_read_bytes(&ch, 1, pdMS_TO_TICKS(200));
        if (n <= 0) {
            // Remind every 60 s of silence, in case the monitor connected late.
            if (esp_timer_get_time() - last_input > 60LL * 1000 * 1000) {
                printf("(type h and Enter for help)\n");
                fflush(stdout);
                last_input = esp_timer_get_time();
            }
            continue;
        }
        last_input = esp_timer_get_time();
        if (ch >= '0' && ch <= '9') {
            const int index = ch - '0';
            if (index < static_cast<int>(g.set.header->n_boards)) {
                g.selected = index;
                ESP_LOGI(TAG, "selected board %d (%s, %s) -- place its sheet under the camera, "
                              "then type c", index, g.set.meta[index].name,
                         g.set.meta[index].is_clean ? "clean" : "defective");
            }
            continue;
        }
        switch (ch) {
        case 'c': command_capture(); break;
        case 'p': command_preview(); break;
        case 'v': command_view(); break;
        case 'w': command_flat_field(); break;
        case 'x':
            g.flat_valid = false;
            ESP_LOGI(TAG, "flat field cleared; using automatic light correction");
            break;
        case 'r': command_resolution(); break;
        case 'b': run_benchmark(); break;
        case 'e': command_energy(); break;
        case 'h': print_help(); break;
        default: break;   // Enter, spaces, unknown keys
        }
    }
}
