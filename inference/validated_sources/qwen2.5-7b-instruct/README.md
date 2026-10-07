# Qwen2.5-7B-Instruct validated inference source

This directory preserves the source and repaired data used for the validated
Qwen GGUF smoke and 800-row full run. It scores complete option-label sequences
with the embedded Qwen chat template and the pinned low-level `llama.cpp`
helper. Model weights and compiled binaries are excluded.

Pins and recorded model-file hashes are in the repository-level
`configs/models.json` and `provenance/MODEL_PROVENANCE.md`.
