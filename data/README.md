# Data

`dataset_v2.json.gz` is the repaired analysis dataset. Decompression yields the
exact `dataset_v2.json` used by all four model runs.

`country_pairs_v2.json` contains 200 directed contrasts corresponding to 100
reciprocal population pairs. `data_repair_summary.json` records the repair and
exclusion counts. `exclusion_summary.csv` presents those records in tabular
form and explicitly marks exclusions for which row-level identifiers were not
available in the source summary.

Validated counts:

- source questions: 1,527
- repaired questions: 1,038
- repaired question-population distributions: 18,801
- directed contrasts: 200
- Jensen--Shannon base: 2

The processed data are included for private preservation and reproducibility.
Review redistribution terms before making the repository public.
