# Mistral-7B-Instruct-v0.3 repaired source handoff

This source-only package repairs the Mistral embedded-template/BOS contract.
It scores the actual embedded-template path with exactly one BOS and requires
token-ID equality to the pinned canonical serialization before any inference.

No GGUF weights, compiled helper, smoke output, full output, synthetic result,
or legacy result is included. Follow `RUNBOOK.md` in a model-capable workspace.

Scientific status at packaging time: source/data gate PASS; genuine repaired
smoke and 800-row full run still pending.
