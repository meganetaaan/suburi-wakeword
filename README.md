# suburi-wakeword

`suburi-wakeword` は、Stack-chan 向けの日本語ウェイクワード **「ハイ、ｽﾀｯｸﾁｬﾝ」** を自力で学習・評価するための実験リポジトリです。

## 方針

- 推論ターゲット: ESP32-S3
- 学習方式: **microWakeWord** で TensorFlow Lite Micro 向け `.tflite` モデルを作る
- ESP32-S3 統合: 最終的に `esp-tflite-micro` で Stack-chan firmware 側に組み込む
- TTS: 無料でローカル実行できる local TTS エンジンを使う
  - 第一候補: **Piper** + `piper-sample-generator`
  - 日本語 voice が不足する場合は Coqui TTS / Style-Bert-VITS2 等を比較し、生成音声のライセンスを確認してから採用する
- 人の音声（human voice）: Web アプリ越しに同意付きで収集する
- 音声データ: Git には入れない。`data/raw/`, `data/processed/`, `artifacts/` は `.gitignore` 済み

## 初期スコープ

1. Web 音声収集アプリを作る
   - 同意画面
   - ブラウザ録音
   - 「ハイ、ｽﾀｯｸﾁｬﾝ」の複数回収録
   - 話者属性は最小限・任意
   - 音声とメタデータを分離して保存
2. ローカル TTS 生成パイプラインを作る
   - positive samples: 「ハイ、ｽﾀｯｸﾁｬﾝ」の発話揺れ
   - hard negative samples: 似た語・部分語・雑談
   - augmentation: noise / reverb / speed / gain
   - current smoke runner: `training/pipeline` uses Piper Plus Tsukuyomi-chan to generate a tiny ignored local dataset, augmentation variants, train/validation/holdout split, threshold sweep, and microWakeWord handoff artifacts
3. microWakeWord 学習パイプラインを作る
   - `.tflite` export at `artifacts/model/stream_state_internal_quant.tflite`
   - manifest JSON at `artifacts/model/hai_stackchan_ja.json`
   - threshold / sliding window の評価
4. ESP32-S3 実機評価
   - FAR: false accepts per hour
   - FRR: false reject rate
   - latency / RAM / CPU

## 重要な境界

- このリポジトリは **コードと設計文書**を管理する。
- 収集した人の音声は原則として公開しない。
- 音声データの保管先・削除手順・同意文面は実装前に明文化する。

## 現在の計画

- [docs/plans/2026-05-11-wakeword-pipeline-plan.md](docs/plans/2026-05-11-wakeword-pipeline-plan.md)
- [docs/design/training-pipeline.md](docs/design/training-pipeline.md)
- [docs/design/web-voice-collection.md](docs/design/web-voice-collection.md)

## 検証

```bash
pnpm install
pnpm run ci
```
