idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    EMBED_FILES
        "model/stream_state_internal_quant.tflite"
        "model/hai_stackchan_ja.json"
)
