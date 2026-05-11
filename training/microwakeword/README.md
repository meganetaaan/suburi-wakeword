# microWakeWord training handoff

This directory pins the shape of the real microWakeWord trainer integration.

Current repository state:

1. `training/pipeline` creates a normalized/split/augmented dataset manifest.
2. The smoke runner exports the same handoff paths expected from microWakeWord:
   - `artifacts/model/stream_state_internal_quant.tflite`
   - `artifacts/model/hai_stackchan_ja.json`
   - `artifacts/metrics/threshold-sweep.json`
3. The `.tflite` produced by the smoke runner is a placeholder contract artifact. Replace `suburi_wakeword.microwakeword.train_microwakeword_smoke_model` with the pinned upstream microWakeWord training command once dependency pinning is stable.

Do not mix English `Hi, Stack-chan` into positive samples for the Japanese baseline. Keep it in holdout/near-miss until the Japanese-only baseline has FRR/FAR numbers.
