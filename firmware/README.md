# firmware/

ESP-IDF project for the Seeed XIAO ESP32-S3 Sense. Nothing here yet.

Two modes, per the project plan:

* **Benchmark mode** — all reported numbers. Load a test/template pair from microSD into PSRAM,
  run the 25 tiles through gate → classifier, and log per-stage latency
  (`esp_timer_get_time()`), memory (`heap_caps_get_minimum_free_size()`) and predictions over
  serial. Also run the 500 test-set templates as defect-free boards, since DeepPCB's defect
  density is unrealistically high and the gate's savings only show on mostly-clean data.
* **Live demo mode** — camera → grayscale → crop → Otsu binarization → align to the stored
  template by a small x/y shift search minimizing XOR count → the same tile pipeline → results
  over WiFi. Live accuracy is not a reported metric.

Note from `eval/gate_analysis.py`: the classical gate should count **eroded** XOR pixels, not
raw ones. A *k*×*k* binary erosion is a sliding AND, separable into two 1-D passes and free of
multiplies, so it is cheap here — and it is what makes the gate work at all.
