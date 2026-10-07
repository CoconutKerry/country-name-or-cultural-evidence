# Qwen GGUF full-run audit report

## Outcome

**PASS.** The authorized Qwen-only CPU run completed exactly 200 cleaned reciprocal directed experimental units under all four conditions, producing 800 unique valid rows. No synthetic or fallback backend was accepted, and no Llama, Mistral, or Gemma run was started.

- Model: `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`
- Hugging Face repository: `Qwen/Qwen2.5-7B-Instruct-GGUF`
- Pinned model revision: `bb5d59e06d9551d752d08b292a50eb208b07ab1f`
- Backend / quantization: `llama.cpp` / `Q4_K_M`
- llama.cpp commit: `62acc89c26c66076cb72e049f307fbe93b8b9750`
- Embedded chat-template SHA-256: `d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`
- Run fingerprint: `fa0e7206f0d622bbf90407155855d6d492ca5353eeca63deecd573055defc9d5`
- Conditions: `baseline`, `country_label`, `evidence`, `conflict`
- Directed units / rows: 200 / 800
- Unique directed-condition keys: 800
- Target units for CI/EI: 144
- Question-ID clusters: 44
- Inference time: 2,548.52 seconds

The model ran on CPU (`gpu_layers = 0`) using eight threads, full-vocabulary low-level candidate-label logits, full conditional sequence scoring for multi-token labels, and the chat template embedded in the official GGUF metadata. Generated-answer parsing was disabled.

## Schema and metric repairs

- The primary signed Evidence Override is explicitly stored as `EO_raw = JSD2(conflict, label) - JSD2(conflict, evidence)`.
- The secondary normalized metric is stored as `EO_normalized = [sqrt(JSD2(conflict, label)) - sqrt(JSD2(conflict, evidence))] / sqrt(JSD2(label, evidence))`.
- Positive Evidence Override is evidence-side; negative is label-side.
- `base2_jensen_shannon_divergences` replaces the misleading distance field name.
- `presented_evidence_distribution` and `source_evidence_distribution` are explicit nulls when evidence is not presented. Human label/evidence reference distributions remain separate for metric computation.
- The full directed key `(question_id, label_country, evidence_country)` is preserved.
- One deterministic option permutation per target unit is shared across all four conditions and is restored to original semantic option order before metrics are computed.

## Statistical results

The bootstrap sampled `question_id` clusters, not individual condition rows. Each of 10,000 seeded replicates retained every selected question's target units, directed pairs, reciprocal directions, four condition contributions, and model rows.

| Metric | N | Estimate | Percentile 95% CI |
|---|---:|---:|---:|
| Country Influence | 144 | 0.028082 | [-0.007666, 0.065957] |
| Evidence Influence | 144 | 0.267590 | [0.220195, 0.319668] |
| EO_raw | 200 | 0.638422 | [0.571393, 0.695376] |
| EO_normalized | 200 | 0.668564 | [0.591620, 0.739419] |

Conflict classification from `EO_raw` at tolerance `1e-12`:

| Class | Count | Percent |
|---|---:|---:|
| evidence-side | 189 | 94.5% |
| label-side | 11 | 5.5% |
| tie | 0 | 0.0% |

## Validation

- Runner completion gate: PASS
- Exact 800-row / 200-unit cardinality: PASS
- Directed key and reciprocal preservation: PASS
- Four-condition atomic checkpoints: PASS
- Fixed permutation and semantic recovery: PASS
- Candidate-label tokenization and full-sequence scoring: PASS
- Candidate sequences independently checked at the exact answer position: 2,592
- Same-first-word semantic option pairs independently checked: 448; exact equal scores/probabilities: 0/0
- Finite, non-negative, normalized probabilities: PASS
- Null evidence-field semantics: PASS
- EO endpoint orientation tests (`+1` evidence, `-1` label): PASS
- Clean national/current-national sample contract: PASS
- 10,000 question-clustered bootstrap: PASS
- Full repository unit suite: 100/100 PASS
- Full four-model approval gate integrity checks: PASS

An independent read-only reproduction matched all result cardinalities, classifications, point estimates, and 10,000 bootstrap confidence intervals. It also rehashed both model shards and reproduced the expected runtime identities.

## Runtime model files

| GGUF shard | Size (bytes) | SHA-256 |
|---|---:|---|
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3,993,201,344 | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689,872,288 | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |

The packaged repository records the model/runtime provenance and checksums but intentionally excludes the multi-gigabyte model weights and compiled binaries.

## Output locations

Raw and tabular outputs are in `experiments/qwen_gguf_full/`. The detailed generated experiment report is `experiments/qwen_gguf_full/EXPERIMENT_REPORT.md`; the complete inference rows are `results.jsonl`; `analysis.json` contains all validated metrics and bootstrap metadata; CSV and LaTeX tables are included beside them.
