# Full inference reproduction

The analysis release does not redistribute model weights. Obtain each GGUF from
the repository and immutable revision recorded in `configs/models.json`, then
verify every file size and SHA-256 before inference. Use a clean checkout of
`llama.cpp` commit `62acc89c26c66076cb72e049f307fbe93b8b9750`.

Each directory under `validated_sources/` contains a model-specific `README.md`
and `RUNBOOK.md`. The separate implementations preserve the chat-template,
BOS, candidate-token, checkpoint, and smoke-test contracts validated for that
model. A typical full reproduction has four stages:

1. install the model-specific inference requirements;
2. compile the low-level GGUF scoring helper against the pinned `llama.cpp`;
3. pass the genuine reciprocal smoke test; and
4. run the 800-row inference entry point and its model-specific analysis.

The four source suites can be checked without model weights by running:

```bash
make validate-inference
```

The released run records were produced on CPU with 4,096-token contexts and
zero GPU layers. Thread counts and runtime fingerprints are listed in
`results/summary/model_runtime.csv` and `provenance/MODEL_PROVENANCE.md`.
