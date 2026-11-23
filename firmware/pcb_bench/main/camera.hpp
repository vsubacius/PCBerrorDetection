// Camera on the Seeed XIAO ESP32-S3 Sense (OV2640 or OV3660 on the B2B
// connector), in grayscale. Thin wrapper over espressif/esp32-camera.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace pcb {
namespace camera {

struct Size {
    int w;
    int h;
    const char *name;
};

// Start the camera. Buffers are sized for the largest mode in sizes().
bool init();
const char *sensor_name();

// Resolutions the live mode can cycle through.
int num_sizes();
Size size(int index);
bool set_size(int index);
int current_size();

// Capture one fresh grayscale frame and copy it into `dst` (w*h bytes,
// caller-owned, at least the largest size). Older frames are discarded first,
// so the frame is taken after this call starts.
bool capture(uint8_t *dst, int *w, int *h);

// Live view: start the camera in JPEG mode at the current resolution and call
// on_frame(jpeg, bytes, w, h) for every frame until should_stop() is true or
// on_frame returns false. The camera is off again afterwards. Returns the
// number of frames, or -1 if the camera did not start.
int stream_jpeg(bool (*on_frame)(const uint8_t *, size_t, int, int), bool (*should_stop)());

}  // namespace camera
}  // namespace pcb
