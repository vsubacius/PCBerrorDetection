# export/

Quantization and conversion to an MCU runtime. Nothing here yet.

Open decision: **ESP-DL (ESP-PPQ quantizer) vs. TFLite Micro (esp-nn)**. Check the current docs
before committing to one — both moved recently. The first task is not a real model: get a
*dummy* int8 model running on the XIAO ESP32-S3 end to end, so the toolchain risk is paid off
early.

Planned: QAT export, PTQ with a calibration set (`TileData.flat_tiles()` gives tiles in the
right layout), and optional structured channel pruning on Stage 2 if it misses the latency
budget.
