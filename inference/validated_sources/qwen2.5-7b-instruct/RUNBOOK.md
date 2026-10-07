# Qwen full-run procedure

Set `MODEL_DIR` to a directory containing both pinned Q4_K_M GGUF shards and
`LLAMA_CPP_DIR` to a clean checkout of the pinned `llama.cpp` commit.

```bash
python scripts/build_gguf_score_helper.py \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --output build/gguf-smoke/gguf_score_helper

python scripts/run_genuine_qwen_gguf_smoke.py \
  --model-dir "$MODEL_DIR" \
  --helper build/gguf-smoke/gguf_score_helper \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --output-dir experiments/smoke_genuine_qwen_gguf

python scripts/run_qwen_gguf_full.py \
  --model-dir "$MODEL_DIR" \
  --helper build/gguf-smoke/gguf_score_helper \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --output-dir experiments/qwen_gguf_full \
  --allow-qwen-full-run

python scripts/analyze_qwen_gguf_full.py \
  --results experiments/qwen_gguf_full/results.jsonl \
  --manifest data/pairs/country_pairs_v2.json \
  --output-dir experiments/qwen_gguf_full \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 42
```

Stop on any failed smoke assertion, model hash mismatch, template mismatch, or
row-count mismatch. Do not substitute a different model revision or
quantization.
