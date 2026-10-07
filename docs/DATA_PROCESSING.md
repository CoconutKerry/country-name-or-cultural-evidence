# Data processing record

The starting export contained 1,527 subjective or value-relevant questions.
The repaired dataset retains 1,038 questions and 18,801 question-population
distributions. The repair procedure canonicalizes option keys, removes
unambiguous missing/refused categories, retains substantive survey responses,
excludes incompatible response schemas and unresolved items, and requires an
exact shared substantive option schema for each population pair.

For every eligible question, base-2 Jensen--Shannon divergence is calculated
between population distributions. At most the three most divergent pairs per
question are retained, followed by selection of the 100 highest-divergence
pairs overall. Evaluating both directions produces 200 directed contrasts over
44 questions and 55 national or territorial sample labels.

The resulting subset is designed to maximize diagnostic resolution for cue
arbitration. `data/data_repair_summary.json` is the machine-readable source for
repair and exclusion counts.
