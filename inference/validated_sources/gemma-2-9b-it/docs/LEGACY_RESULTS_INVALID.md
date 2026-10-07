# Legacy Result Validity Notice

All files under `experiments/main/`, the dated files under
`experiments/pilot/`, and the pre-existing files under `analysis/output/` were
generated before the 2026-08-30 repair. They are preserved for forensic
comparison only.

They are invalid for scientific use because:

1. predictions were computed from first tokens of answer texts rather than the
   displayed option labels;
2. data included ineligible samples and incompatible answer schemas;
3. analysis collapsed different evidence-source countries; and
4. metric reports mixed JSD units.

Do not relabel, rescale, or partially re-analyze these files as repaired
results. The scorer defect requires fresh inference. New scientific outputs
must contain all of the following metadata:

- `scoring_method: complete_option_label_continuation_log_likelihood`;
- `directed_unit_key: [question_id, country, conflict_country]`;
- `jensen_shannon_base: 2`;
- `synthetic_backend: false`; and
- `scientific_inference: true`;
- a present package-lock identity; and
- per-model chat-template hashes plus matching requested/resolved model and
  tokenizer commit revisions.
