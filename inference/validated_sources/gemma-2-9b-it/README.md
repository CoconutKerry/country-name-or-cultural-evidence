# Gemma-2-9B-It repaired source handoff

This source-only package repairs the Gemma embedded-template/BOS contract. A
serialization-only helper operation records the live embedded expansion and
both tokenization modes without calling `llama_decode`; inference starts only
after the exact one-BOS byte/token-equivalence contract passes.

No GGUF weights, compiled helper, smoke output, full output, synthetic result,
or legacy result is included. The helper must be rebuilt from the repaired C++
source. Follow `RUNBOOK.md` in a model-capable workspace.

Scientific status at packaging time: source/data gate PASS; genuine repaired
smoke and 800-row full run still pending.
