# Mistral repaired-run runbook

Use only the exact pinned GGUF and llama.cpp checkout already available in the
model-capable workspace. Do not substitute a model, quantization, template,
backend, generated-answer parser, synthetic runner, mock, or fallback.

## Pins

- GGUF: `bartowski/Mistral-7B-Instruct-v0.3-GGUF` revision `61fd4167fff3ab01ee1cfe0da183fa27a944db48`
- File: `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`, 4,372,812,000 bytes, SHA-256 `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6`
- Upstream identity: `mistralai/Mistral-7B-Instruct-v0.3`; the quantizer did not attest an upstream conversion revision, so it remains `null`
- llama.cpp commit: `62acc89c26c66076cb72e049f307fbe93b8b9750`

## Run

Set `MODEL_DIR` to the directory containing exactly the pinned GGUF and
`LLAMA_CPP_DIR` to the clean pinned checkout. Then run:

```bash
python scripts/write_source_preinference_gate.py
git -C "$LLAMA_CPP_DIR" rev-parse HEAD
python scripts/build_gguf_score_helper.py \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --output build/gguf-smoke/gguf_score_helper
python scripts/run_genuine_mistral_gguf_smoke.py \
  --model-dir "$MODEL_DIR" \
  --helper build/gguf-smoke/gguf_score_helper \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --test-report audit/PREINFERENCE_GATE.json \
  --smoke-output experiments/mistral_gguf_smoke \
  --full-output experiments/mistral_gguf_full \
  --analysis-output experiments/mistral_gguf_analysis
```

The entrypoint starts the 800-row full run, then the 10,000-replicate analysis,
only after all eight genuine-smoke rows and hard assertions pass. On any
failure, stop and return the diagnostics; do not invoke the full runner
manually.

Return the three output directories together with `audit/PREINFERENCE_GATE.json`
and `audit/PREINFERENCE_TEST_RESULTS.txt` in one ZIP. Do not return model weights.
