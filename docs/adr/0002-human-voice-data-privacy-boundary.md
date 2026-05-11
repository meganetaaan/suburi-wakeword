# ADR 0002: Keep human voice data private and outside Git

## Status

Accepted

## Context

The project needs human recordings of **「ハイ、ｽﾀｯｸﾁｬﾝ」** to evaluate and improve the wake word. Voice recordings are personal data. The code repository can be public, but raw voice data should not be committed or exposed accidentally.

## Decision

- Store raw and processed audio outside Git-tracked paths or under ignored directories.
- Keep `data/raw/`, `data/processed/`, `artifacts/`, `runs/`, and private models ignored.
- Require explicit consent before recording.
- Keep speaker metadata minimal and optional.
- Separate contributor identity/contact information from audio files whenever possible.
- Provide deletion/export procedures before inviting outside participants.

## Consequences

- Public collaboration remains possible without exposing recordings.
- Training runs must reference a local or private object-storage dataset path.
- Reproducibility must be achieved through manifests, hashes, and scripts, not by committing raw data.
