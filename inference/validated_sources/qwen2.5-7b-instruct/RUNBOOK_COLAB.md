# Google Colab GPU Handoff Runbook

This runbook accompanies `cultural_alignment_audit_colab.ipynb`. It starts
from a fresh Google Colab session and does not use any path from the Work
environment in which the repository was repaired.

## Safety and approval boundary

The completed deterministic smoke test validates pipeline plumbing only. It
does **not** validate real checkpoint access, tokenization, chat templates,
option-label likelihoods, 4-bit quantization, CUDA compatibility, or GPU memory.

Run the genuine Qwen smoke and inspect its assertions and saved records first.
The four-model cells are deliberately disabled. Do not enable or execute them
until the genuine smoke has passed and explicit approval for the full run has
been given. Never substitute a different checkpoint when access fails.

## 1. Start a fresh Colab session

1. Open <https://colab.research.google.com/>.
2. Extract the handoff ZIP on your computer. Choose **File → Upload
   notebook** and select `cultural_alignment_audit_colab.ipynb` from the
   extracted folder. Keep the original handoff ZIP intact for the repository
   setup cell.
3. Choose **Runtime → Change runtime type**.
4. Select **Python 3** and a **GPU** hardware accelerator. A high-memory GPU is
   preferable. The notebook uses 4-bit loading, but actual memory use depends
   on the GPU, CUDA runtime, checkpoint, sequence lengths, and Colab image.
5. Choose **Runtime → Disconnect and delete runtime** before a clean rerun if
   the session has previously had incompatible packages or models loaded.

Do not proceed past the environment cell unless CUDA is available and the
reported GPU memory is adequate for the selected checkpoint. The notebook
reports Python, RAM, disk, GPU model, CUDA version, and GPU memory; retain this
record with the results.

## 2. Make the repaired repository available

Use one of the notebook's two supported input methods. Do not enter a path from
the Work environment.

### Option A: upload the handoff ZIP

Run the upload cell and select the supplied Colab handoff ZIP. The notebook
validates and extracts it into the current Colab runtime, then discovers the
repository by the presence of `cultural_alignment_audit_colab.ipynb`,
`requirements-colab.lock`, `src/`, `tests/`, and `data/`.

The uploaded runtime filesystem is ephemeral. Persist output to Google Drive
or download it before the session ends.

### Option B: mount Google Drive

1. Upload the handoff ZIP to a folder in your own Drive.
2. Run the Drive-mount cell and authorize access.
3. Set the notebook's ZIP input variable to that Drive file.
4. Set its output root to a separate Drive folder if results should survive a
   Colab disconnect.

Use a new, empty output directory for a scientifically distinct run. A Drive
directory is strongly recommended for `OUTPUT_ROOT`, because it preserves each
per-unit checkpoint across Colab runtime loss. Reuse the same output directory
only when intentionally resuming that run.

## 3. Install and verify pinned dependencies

Run the dependency-install cell. It installs the exact direct pins in
`requirements-colab.lock`, including the Transformers, Accelerate,
Hugging Face Hub, and 4-bit `bitsandbytes` stack.

If Colab asks for a runtime restart after installation, restart and then rerun
the notebook from the first configuration code cell. Do not continue with a partially
reloaded Python process. Preserve the emitted package-version manifest with the
results.

Next, run the environment-report cell and confirm all of the following:

- `torch.cuda.is_available()` is true;
- a real GPU model and nonzero GPU memory are reported;
- PyTorch has a CUDA version;
- `bitsandbytes` imports successfully; and
- sufficient disk remains for the Qwen checkpoint/cache plus smoke outputs.
  Before any later approved full run, budget for the cumulative Hugging Face
  cache footprint of all four checkpoints plus output/checkpoint files. CUDA
  cleanup releases device memory but does not delete downloaded model files.

The notebook loads only one model at a time and deletes model/tokenizer objects,
runs garbage collection, and empties the CUDA cache before loading another.
That cleanup does not increase the physical memory of the selected GPU.

## 4. Supply a Hugging Face token securely

Some configured checkpoints are gated. Obtain access under the same Hugging
Face account before running the access check.

Preferred Colab method:

1. Open the **Secrets** panel (key icon) in the left sidebar.
2. Add a secret named `HF_TOKEN` containing a read-only Hugging Face token.
3. Enable notebook access for that secret.

Alternative method: arrange for `HF_TOKEN` to be present in the process
environment before running an access or inference cell. Do not paste a token
into a notebook source cell, output, configuration file, result, archive, or
version-control history. The notebook reads the secret at runtime and does not
write its value to disk.

The access-check cell checks these exact model IDs and pinned revisions:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it`

An inaccessible or gated model is reported. The full-run preflight then aborts
before loading any full-run weights. Resolve access with the model publisher
and rerun the check; do not silently replace it with another repository,
revision, or model. The preflight probes the exact revision's configuration,
tokenizer configuration, and one declared weight artifact (or weight index),
not merely its public repository metadata.

## 5. Run all tests before inference

Run the unit-test cell before loading any model. It executes the complete suite:

```bash
python -m unittest discover -s tests -v
```

Inference must remain blocked if any test fails. Save the complete test output.
The repaired data and directed-pair validators should also pass before the
genuine smoke begins.

## 6. Run the genuine Qwen smoke

The enabled smoke configuration is fixed to:

- model: `Qwen/Qwen2.5-7B-Instruct` at its pinned revision;
- backend: genuine Hugging Face model inference with 4-bit quantization;
- units: the two reciprocal directed units in
  `data/pairs/country_pairs_smoke.json`;
- conditions: `baseline`, `country_label`, `population_evidence`, and
  `conflict`; and
- scorer: full option-label continuation likelihoods from the repaired
  implementation, mapped back to the original semantic options.

Run the access check first, then the genuine-smoke cell. Eight prediction rows
are expected: two directed units × four conditions. The notebook appends and
flushes each completed experimental unit to an atomic checkpoint under
`<OUTPUT_ROOT>/smoke/checkpoints/qwen2.5-7b-instruct/`, then rebuilds the
convenience JSONL result. On restart, it validates the existing checkpoints and
skips only complete, valid keys; it does not infer completion from a partially
written unit.

For every prediction, inspect that the saved row contains:

- `question_id`, `label_country`, `evidence_country`, and `condition`;
- the original semantic options and displayed option labels;
- the full chat-templated prompt;
- tokenizer IDs for every candidate label;
- raw label log-probabilities and normalized semantic-option probabilities;
- both countries' human distributions;
- base-2 Jensen–Shannon distances; and
- Country Influence, Evidence Influence, and Evidence Override. In a
  condition-row JSONL representation, the directed unit's derived influence
  values may be repeated on each of its four rows, but the fields must be
  present in every saved prediction.

The smoke assertion report must pass all checks for answer-position scoring,
valid label tokenization, same-prefix semantic-option separation, finite and
normalized probabilities, permutation recovery, directed-key uniqueness, and
the sign convention for Evidence Override. The sign convention is:

```text
Evidence Override = JSD(prediction, label distribution)
                  - JSD(prediction, evidence distribution)
```

It must be positive when the prediction equals the evidence distribution and
negative when it equals the label distribution (for distinct distributions).

Do not treat a completed smoke as sufficient merely because eight rows exist.
Confirm the model ID and pinned revision, genuine backend, 4-bit load metadata,
chat-template prompt, candidate token IDs, raw log-probabilities, probability
sums, assertion status, and absence of synthetic-backend metadata.

## 7. Resume safely after interruption

1. Reopen the same notebook and select a GPU runtime.
2. Remount Drive or upload the same handoff ZIP.
3. Point the notebook to the **same output directory** used by the interrupted
   run.
4. Reinstall dependencies, rerun the environment report, rerun the complete
   tests, restore `HF_TOKEN`, and rerun the model-access check.
5. Run the same smoke or approved full-run cell. The resume logic validates the
   saved directed key and condition records, skips complete units, and starts
   with the first incomplete unit.

Do not hand-edit JSONL checkpoints to force completion. If validation reports a
duplicate, malformed, inconsistent, or partial unit, preserve the file for
diagnosis and resume into a new output directory after correcting the cause.
Never combine rows from different checkpoint revisions, dependency manifests,
pair manifests, or run configurations.

## 8. Inspect before any full run

Stop after the genuine Qwen smoke. Review its environment report, tests, access
report, records, and assertion report. The full-run control must remain false
or disabled until explicit approval is supplied after that review. The full
cell ships with `FULL_RUN_APPROVED = False` and a separate exact approval-phrase
check. Only after approval, set the boolean to true and enter exactly
`I APPROVE THE FULL FOUR-MODEL RUN` in the approval-phrase variable. Both gates
must remain unchanged before approval.

After approval, the full-run configuration is fixed to the cleaned 200
reciprocal directed units in `data/pairs/country_pairs_v2.json`, all four
conditions, and the four exact models listed above. The complete unit identity
must remain:

```text
(question_id, label_country, evidence_country)
```

Models run sequentially, never concurrently. After each model, the notebook
saves and validates its output, deletes the model and tokenizer, performs
Python garbage collection, and empties CUDA memory before the next model.

A model is complete only when the completion report confirms:

- exactly 200 unique directed keys for that exact model and revision;
- both directions of every reciprocal pair are present;
- exactly the four required conditions exist for every directed key;
- therefore exactly 800 valid prediction rows are present;
- there are no duplicate or unexpected keys or conditions;
- every probability vector and semantic permutation validates; and
- the saved data, pair-manifest, code/config, and runtime identities match the
  run manifest.

An access failure, out-of-memory error, interrupted write, failed assertion, or
row-count mismatch means the model is not complete.

## 9. Statistical analysis

After all explicitly approved model runs are complete, run the analysis cell.
It performs 10,000 percentile-bootstrap replicates clustered by `question_id`.
Each replicate samples question IDs with replacement and retains all directed
pairs, reciprocal directions, conditions, and models belonging to every
sampled question. It does not bootstrap condition rows independently.

The analysis reports per-model point estimates and percentile 95% confidence
intervals for Country Influence, Evidence Influence, and Evidence Override.
Confirm that the output metadata records 10,000 replicates, the random seed,
`question_id` as the cluster, and the participating model revisions.

## 10. Expected outputs

The notebook creates a run-specific `OUTPUT_ROOT`. Preserve these files:

| Path under `OUTPUT_ROOT` | Purpose |
|---|---|
| `environment.json` | Python, RAM, disk, GPU, CUDA, GPU memory, and package versions |
| `unit_tests.txt` | Complete pre-inference test output |
| `dependency_install.txt`, `pip_check.txt`, `package_freeze.txt` | Install transcript, Colab-image compatibility diagnostics, and full installed-package snapshot |
| `model_access.json` | Accessibility of every exact checkpoint and pinned revision |
| `smoke/checkpoints/qwen2.5-7b-instruct/<unit-hash>.json` | Atomic per-directed-unit Qwen smoke checkpoint containing all four conditions |
| `smoke/results_qwen2.5-7b-instruct.jsonl` | Eight validated Qwen smoke prediction rows |
| `smoke/smoke_assertions.json` | Machine-readable pass/fail evidence for every required smoke assertion |
| `smoke/run_signature_*.json`, `smoke/model_runtime_*.json`, `smoke/completion_*.json` | Resume fingerprint, immutable runtime provenance, and strict per-model completion report |
| `full/checkpoints/<model-slug>/<unit-hash>.json` | Atomic per-directed-unit checkpoint containing all four conditions for an approved model |
| `full/results_<model-slug>.jsonl` | Exactly 800 validated rows when that approved model is complete |
| `full/run_signature_*.json`, `full/model_runtime_*.json`, `full/completion_*.json` | Per-model resume fingerprint, runtime provenance, and strict completion report |
| `analysis/clustered_bootstrap.json` | Per-model estimates and 10,000-replicate percentile 95% intervals |
| `cultural_alignment_audit_results.zip` | Downloadable result package produced after validation |

The notebook also records run/data/model identities within results and reports.
The supplied synthetic plumbing outputs remain under `experiments/smoke/` and
must not be merged with, renamed as, or analyzed as genuine model output.

## 11. Download and package final results

Use the final packaging cell only after the desired outputs validate. It should
copy—not mutate—the results, reports, manifests, notebook, runbook, dependency
lock, repaired source/tests, and relevant pair/data manifests into one archive,
then print a SHA-256 checksum.

Download the ZIP from Colab's Files pane, or retain it in the configured Drive
output folder. If the packaging cell emits a checksum file, download that too.
Verify locally with:

```bash
unzip -t cultural_alignment_audit_results.zip
sha256sum cultural_alignment_audit_results.zip
```

Before sharing, inspect the archive file list and search for accidental
credentials. It must not contain `HF_TOKEN`, notebook secrets, Hugging Face
cache files, downloaded model weights, or other account-specific material.
