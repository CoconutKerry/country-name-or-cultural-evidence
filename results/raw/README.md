# Raw model outputs

Each model directory contains the exact 800-row `results.jsonl` compressed with
gzip, together with its runtime, completion, run-signature, and original
analysis records. Decompression does not alter the JSONL bytes.

The result rows retain complete option probabilities, human reference
distributions, fixed permutations, candidate sequence scores, prompt hashes,
model identifiers, and runtime fingerprints. Generated-answer parsing was not
used and every row is marked `synthetic: false`.
