#include "camera.hpp"

#include <string.h>

#include "esp_camera.h"
#include "esp_log.h"

namespace pcb {
namespace camera {

namespace {

const char *TAG = "camera";

// XIAO ESP32-S3 Sense camera pins (Seeed's CAMERA_MODEL_XIAO_ESP32S3).
constexpr int kPinPwdn = -1;
constexpr int kPinReset = -1;
constexpr int kPinXclk = 10;
constexpr int kPinSda = 40;
constexpr int kPinScl = 39;
constexpr int kPinD7 = 48;
constexpr int kPinD6 = 11;
constexpr int kPinD5 = 12;
constexpr int kPinD4 = 14;
constexpr int kPinD3 = 16;
constexpr int kPinD2 = 18;
constexpr int kPinD1 = 17;
constexpr int kPinD0 = 15;
constexpr int kPinVsync = 38;
constexpr int kPinHref = 47;
constexpr int kPinPclk = 13;

struct Mode {
    framesize_t frame;
    Size size;
};

const Mode kModes[] = {
    {FRAMESIZE_SVGA, {800, 600, "SVGA 800x600"}},
    {FRAMESIZE_XGA, {1024, 768, "XGA 1024x768"}},
    {FRAMESIZE_UXGA, {1600, 1200, "UXGA 1600x1200"}},
};
constexpr int kNumModes = sizeof(kModes) / sizeof(kModes[0]);

int g_mode = 0;
constexpr int kSettleFrames = 8;
char g_sensor[24] = "none";

}  // namespace

// Start the driver at one size. In grayscale (non-JPEG) mode the driver drops
// every frame whose size differs from the size it was started with
// (cam_hal.c, "FB-SIZE"), so a size change needs a full restart; changing the
// size with set_framesize() alone makes every capture time out.
bool start(int index, bool jpeg = false)
{
    camera_config_t config = {};
    config.pin_pwdn = kPinPwdn;
    config.pin_reset = kPinReset;
    config.pin_xclk = kPinXclk;
    config.pin_sccb_sda = kPinSda;
    config.pin_sccb_scl = kPinScl;
    config.pin_d7 = kPinD7;
    config.pin_d6 = kPinD6;
    config.pin_d5 = kPinD5;
    config.pin_d4 = kPinD4;
    config.pin_d3 = kPinD3;
    config.pin_d2 = kPinD2;
    config.pin_d1 = kPinD1;
    config.pin_d0 = kPinD0;
    config.pin_vsync = kPinVsync;
    config.pin_href = kPinHref;
    config.pin_pclk = kPinPclk;
    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.pixel_format = jpeg ? PIXFORMAT_JPEG : PIXFORMAT_GRAYSCALE;
    config.frame_size = kModes[index].frame;
    // Live view: small JPEG frames, two buffers, always the newest frame.
    config.jpeg_quality = 16;
    config.fb_count = jpeg ? 2 : 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = jpeg ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;

    const esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: %s (is the camera board plugged in?)",
                 esp_err_to_name(err));
        return false;
    }
    sensor_t *s = esp_camera_sensor_get();
    if (s != nullptr) {
        const camera_sensor_info_t *info = esp_camera_sensor_get_info(&s->id);
        snprintf(g_sensor, sizeof(g_sensor), "%s (PID 0x%x)",
                 info != nullptr ? info->name : "unknown", s->id.PID);
    }
    g_mode = index;
    return true;
}

// The camera is only running during capture(). Between captures it is off,
// so its driver task prints nothing while results and images are sent over
// USB (its warnings bypass the normal log and can land inside image data).
bool init()
{
    if (!start(0)) {
        return false;
    }
    esp_camera_deinit();
    return true;
}

const char *sensor_name()
{
    return g_sensor;
}

int num_sizes()
{
    return kNumModes;
}

Size size(int index)
{
    return kModes[index].size;
}

int current_size()
{
    return g_mode;
}

bool set_size(int index)
{
    if (index < 0 || index >= kNumModes) {
        return false;
    }
    const int previous = g_mode;
    if (start(index)) {
        esp_camera_deinit();
        return true;
    }
    ESP_LOGE(TAG, "could not start the camera at %s; staying at %s",
             kModes[index].size.name, kModes[previous].size.name);
    g_mode = previous;
    return false;
}

bool capture(uint8_t *dst, int *w, int *h)
{
    if (!start(g_mode)) {
        return false;
    }
    // The first frames after start-up are dark or badly exposed while the
    // automatic exposure settles: drop them.
    for (int i = 0; i < kSettleFrames; ++i) {
        camera_fb_t *old = esp_camera_fb_get();
        if (old != nullptr) {
            esp_camera_fb_return(old);
        }
    }
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == nullptr) {
        ESP_LOGE(TAG, "no frame from the camera");
        esp_camera_deinit();
        return false;
    }
    const size_t n = static_cast<size_t>(fb->width) * fb->height;
    bool ok = fb->format == PIXFORMAT_GRAYSCALE && fb->len >= n;
    if (ok) {
        memcpy(dst, fb->buf, n);
        *w = fb->width;
        *h = fb->height;
    } else {
        ESP_LOGE(TAG, "unexpected frame: format %d, %ux%u, %u bytes", fb->format,
                 static_cast<unsigned>(fb->width), static_cast<unsigned>(fb->height),
                 static_cast<unsigned>(fb->len));
    }
    esp_camera_fb_return(fb);
    esp_camera_deinit();
    return ok;
}

int stream_jpeg(bool (*on_frame)(const uint8_t *, size_t, int, int), bool (*should_stop)())
{
    // Same resolution as capture(), so the view (field of view) is the same.
    if (!start(g_mode, true)) {
        return -1;
    }
    int frames = 0;
    while (!should_stop()) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == nullptr) {
            continue;
        }
        const bool keep_going = on_frame(fb->buf, fb->len, fb->width, fb->height);
        esp_camera_fb_return(fb);
        ++frames;
        if (!keep_going) {
            break;
        }
    }
    esp_camera_deinit();
    return frames;
}

}  // namespace camera
}  // namespace pcb
