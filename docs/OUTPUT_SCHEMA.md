# Output schema

## Raw rows

The gzip files in `results/raw/*/results.jsonl.gz` preserve the original model
output schema. Important fields include:

- model identity: `model_identifier`, `model_repository`, `model_revision`,
  `quantization`, `run_fingerprint`;
- unit identity: `question_id`, `label_country`, `evidence_country`, `unit_id`,
  `target_unit_id`, `condition`;
- option mapping: `original_options`, `displayed_option_order`,
  `displayed_option_labels`, `displayed_label_to_option`, `permutation_seed`;
- predictions: `raw_candidate_scores`, `normalized_label_probabilities`,
  `normalized_prediction`;
- references: `label_country_human_distribution`,
  `evidence_country_human_distribution`, `presented_evidence_distribution`;
- prompt provenance: `raw_user_prompt`, `structured_messages`,
  `serialized_chat_templated_prompt`, `serialized_prompt_sha256`; and
- scoring provenance: `candidate_label_token_ids`, `scoring_trace`,
  `scoring_method`.

## Harmonized rows

`results/processed/predictions_all_models.jsonl.gz` contains a common subset of
fields across the four model-specific schemas. It maps the raw condition name
`evidence` to `population_evidence` and adds the survey source domain.

## Unit-level metrics

`target_unit_metrics.csv` contains one record per model and unique
`(question_id, label_country)` target. It is the analysis unit for CI, EI, and
EI-CI. `directed_unit_metrics.csv` contains one record per model and directed
`(question_id, label_country, evidence_country)` contrast. It is the analysis
unit for Evidence Override and conflict classification.
