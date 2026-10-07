# Llama repaired-run runbook (source package v2)

Use only the exact pinned GGUF and llama.cpp checkout already available in the
model-capable workspace. Do not substitute a model, quantization, template,
backend, generated-answer parser, synthetic runner, mock, or fallback.

## Pins

- GGUF: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` revision `bf5b95e96dac0462e2a09145ec66cae9a3f12067`
- File: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, 4,920,739,232 bytes, SHA-256 `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c`
- Upstream pin: `meta-llama/Meta-Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659`
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
python scripts/run_genuine_llama_gguf_smoke.py \
  --model-dir "$MODEL_DIR" \
  --helper build/gguf-smoke/gguf_score_helper \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --gguf-revision bf5b95e96dac0462e2a09145ec66cae9a3f12067 \
  --upstream-revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --output-dir experiments/smoke_genuine_llama_gguf \
  --full-output-dir experiments/llama_gguf_full
```

The entrypoint starts the 800-row full run only after all eight genuine-smoke
rows and hard assertions pass. On any failure, stop and return the diagnostics;
do not invoke the full runner manually.

After full completion:

```bash
python scripts/analyze_llama_gguf_full.py \
  --results experiments/llama_gguf_full/results.jsonl \
  --manifest data/pairs/country_pairs_v2.json \
  --output-dir experiments/llama_gguf_full \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 42
python scripts/package_llama_gguf_full.py ../llama-3.1-8b-instruct-full-results.zip
```

Return the result ZIP and its SHA-256 sidecar. Do not return model weights.
