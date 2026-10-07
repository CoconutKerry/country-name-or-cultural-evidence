# Full inference sources

The four directories under `validated_sources/` preserve the final model-specific
GGUF pipelines used for the genuine runs. Separate snapshots are retained
because chat-template serialization, BOS handling, and candidate-label token
boundaries were validated independently for each model.

Use `configs/models.json` and `provenance/MODEL_PROVENANCE.md` for the exact
GGUF repositories, immutable revisions, filenames, file sizes, SHA-256 values,
and `llama.cpp` commit. No model weights or compiled executables are included.

Each source directory contains repaired data, low-level scoring-helper source,
pre-inference gates, smoke and full-run entry points, model-specific analysis,
and regression tests. The Qwen directory is a clean snapshot of the final GGUF
path; obsolete v1 experiment outputs and a legacy Colab-only test requiring the
invalidated v1 dataset are omitted. The byte-for-byte authoritative Qwen result
package is preserved in the separate internal archive.

See `FULL_INFERENCE.md` for the common procedure and the `README.md` and
`RUNBOOK.md` inside each model directory for exact commands.
