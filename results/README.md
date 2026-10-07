# Results

This directory separates the preserved inference records from files derived by
the cross-model analysis.

## Layout

- `raw/`: the exact 800-row result stream for each model, compressed with
  deterministic gzip, together with the run's runtime, completion, signature,
  and original analysis records.
- `processed/predictions_all_models.jsonl.gz`: a harmonized 3,200-row master
  file used by the release analysis.
- `processed/target_unit_metrics.csv`: one row for each model and unique
  `(question_id, label_country)` target; 576 rows in total.
- `processed/directed_unit_metrics.csv`: one row for each model and directed
  `(question_id, label_country, evidence_country)` contrast; 800 rows in total.
- `processed/bootstrap_metrics_long.csv`: point estimates and 95% intervals for
  the five reported estimands.
- `summary/`: paper-level metrics, conflict classifications, cross-model
  agreement, source-stratified analyses, selected cases, and the readable
  `RESULTS.md` report.
- `figures/`: regenerated Figure 2 in PDF and PNG formats.
- `latex/`: table rows generated directly from the validated summary files.

Run `make reproduce` from the repository root to regenerate all derived
analysis files and Figure 2 from `raw/`.
