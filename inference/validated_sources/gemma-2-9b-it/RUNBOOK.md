# Gemma repaired-run runbook

Use only the exact pinned GGUF and llama.cpp checkout already available in the
model-capable workspace. Do not reuse the old helper binary. Do not substitute
a model, quantization, template, backend, generated-answer parser, synthetic
runner, mock, or fallback.

## Pins

- GGUF: `bartowski/gemma-2-9b-it-GGUF` revision `d731033f3dc4018261fd39896e50984d398b4ac5`
- File: `gemma-2-9b-it-Q4_K_M.gguf`, 5,761,057,728 bytes, SHA-256 `13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe`
- Current upstream pin: `google/gemma-2-9b-it` revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`; this is not claimed as publisher-attested historical conversion provenance
- llama.cpp commit: `62acc89c26c66076cb72e049f307fbe93b8b9750`

## Run

Set `MODEL_DIR` to the directory containing exactly the pinned GGUF and
`LLAMA_CPP_DIR` to the clean pinned checkout. Use the canonical output paths
below so the result packager can validate them.

```bash
python scripts/write_gemma_preinference_gate.py
git -C "$LLAMA_CPP_DIR" rev-parse HEAD
python scripts/build_gguf_score_helper.py \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --output build/gguf-smoke/gguf_score_helper
python scripts/run_genuine_gemma_gguf_smoke.py \
  --model-dir "$MODEL_DIR" \
  --helper build/gguf-smoke/gguf_score_helper \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --output-dir experiments/smoke_genuine_gemma_gguf \
  --full-output-dir experiments/gemma_gguf_full
```

The serialization-only preflight performs no inference. The entrypoint starts
the 800-row full run only after that preflight and all eight genuine-smoke rows
pass every hard assertion. On any failure, stop and return the diagnostics; do
not invoke the full runner manually.

After full completion:

```bash
python scripts/analyze_gemma_gguf_full.py \
  --results experiments/gemma_gguf_full/results.jsonl \
  --manifest data/pairs/country_pairs_v2.json \
  --output-dir experiments/gemma_gguf_full \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 42
python scripts/package_gemma_gguf_full.py \
  --destination gemma-2-9b-it-full-results.zip
```

The packager revalidates the current source/data dependency fingerprint, smoke
compatibility record, pre-inference gate, runtime gate, and all result rows.
Return the result ZIP and its SHA-256 sidecar. Do not return model weights.
