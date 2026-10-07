# Llama source package v2 packaging correction

Version 1 omitted three files that `scripts/run_genuine_llama_gguf_smoke.py`
hashes before inference to attest that the existing full-run approval gates did
not change during the smoke:

- `src/colab_runner.py`
- `scripts/build_colab_notebook.py`
- `cultural_alignment_audit_colab.ipynb`

The files in version 2 are copied byte-for-byte from the authoritative repaired
Llama repository in `cultural-alignment-audit-three-model-repairs-20260831.zip`.
Their SHA-256 values are:

- `src/colab_runner.py`: `70ae53b6420163bac6b5862df9084ac0dc0933f2c13dfebb1cb85a15196a90ae`
- `scripts/build_colab_notebook.py`: `4cc8274e46fca7ce78c8527e088c588624dd2bfb45961efc535d6ebba39e667d`
- `cultural_alignment_audit_colab.ipynb`: `315124add7b257af66abd1c9ec2120486222d894e009ae4773f7c11601a666f3`

No scoring, model, data, template, metric, smoke, or full-run behavior was
changed. The runbook packaging command was also corrected so the final result
ZIP is written outside the project root, as required by the package validator.
