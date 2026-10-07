# Llama-3.1-8B-Instruct repaired source handoff v2

This source-only package repairs the false-negative Llama chat-template gate.
The pinned template may express BOS through the Jinja `bos_token` variable;
the runner accepts that narrow equivalent while preserving every rendered
prompt, token, answer-position, no-generation, and independent-rescore check.

Version 2 also corrects a source-package omission in version 1. The genuine
smoke entrypoint cryptographically protects three existing approval-gate files;
those files are now included in this package and listed in the source manifest.
No inference logic, scoring implementation, dataset, model pin, or scientific
protocol was changed by this packaging correction.

No GGUF weights, compiled helper, smoke output, full output, synthetic result,
or legacy result is included. Follow `RUNBOOK.md` in a model-capable workspace.

Scientific status at packaging time: source/data gate PASS; genuine repaired
smoke and 800-row full run still pending.
