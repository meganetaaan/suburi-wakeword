#include <stdint.h>
#include <stdio.h>

#include "esp_log.h"

extern const uint8_t stream_state_internal_quant_tflite_start[] asm("_binary_stream_state_internal_quant_tflite_start");
extern const uint8_t stream_state_internal_quant_tflite_end[] asm("_binary_stream_state_internal_quant_tflite_end");
extern const uint8_t hai_stackchan_ja_json_start[] asm("_binary_hai_stackchan_ja_json_start");
extern const uint8_t hai_stackchan_ja_json_end[] asm("_binary_hai_stackchan_ja_json_end");

static const char *TAG = "suburi_wakeword";

void app_main(void) {
    const size_t model_size = (size_t)(stream_state_internal_quant_tflite_end - stream_state_internal_quant_tflite_start);
    const size_t manifest_size = (size_t)(hai_stackchan_ja_json_end - hai_stackchan_ja_json_start);

    ESP_LOGI(TAG, "suburi wakeword smoke app");
    ESP_LOGI(TAG, "model bytes: %u", (unsigned)model_size);
    ESP_LOGI(TAG, "manifest bytes: %u", (unsigned)manifest_size);
    ESP_LOGI(TAG, "TODO: wire microphone + microWakeWord preprocessor + esp-tflite-micro interpreter");
}
