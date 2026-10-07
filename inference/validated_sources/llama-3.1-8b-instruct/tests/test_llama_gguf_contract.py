"""Model-free contract tests for the genuine Llama GGUF pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from scripts import analyze_llama_gguf_full as analysis
from scripts import run_genuine_llama_gguf_smoke as smoke
from scripts import run_llama_gguf_full as full
from src.data_loader import DataLoader
from src.llama_gguf_runner import (
    LLAMA31_ANSWER_SUFFIX,
    LLAMA31_BOS,
    LLAMA31_DEFAULT_SYSTEM_MESSAGE,
    LLAMA31_EOT,
    LLAMA31_HEADER_PREFIX,
    verify_llama31_chat_template,
    verify_llama31_serialized_chat,
)


ROOT = Path(__file__).resolve().parents[1]


class LlamaGGUFContractTests(unittest.TestCase):
    def test_exact_model_provenance_is_pinned(self):
        self.assertEqual(
            smoke.MODEL_REPOSITORY,
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        )
        self.assertEqual(
            smoke.MODEL_REVISION,
            "bf5b95e96dac0462e2a09145ec66cae9a3f12067",
        )
        self.assertEqual(smoke.QUANTIZATION, "Q4_K_M")
        self.assertEqual(len(smoke.MODEL_SHARDS), 1)
        selected = smoke.MODEL_SHARDS[0]
        self.assertEqual(
            selected["filename"],
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        )
        self.assertEqual(selected["size_bytes"], 4_920_739_232)
        self.assertEqual(
            selected["sha256"],
            "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c",
        )

    def test_llama_cpp_and_data_revisions_are_pinned(self):
        self.assertEqual(
            smoke.LLAMA_CPP_COMMIT,
            "62acc89c26c66076cb72e049f307fbe93b8b9750",
        )
        self.assertEqual(
            hashlib.sha256((ROOT / full.PAIR_MANIFEST).read_bytes()).hexdigest(),
            full.PAIR_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((ROOT / full.DATASET_PATH).read_bytes()).hexdigest(),
            full.DATASET_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "src/gguf_runner.py").read_bytes()).hexdigest(),
            full.ACCEPTED_GGUF_RUNNER_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "src/llama_gguf_runner.py").read_bytes()
            ).hexdigest(),
            full.LLAMA_GGUF_RUNNER_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "src/gguf_score_helper.cpp").read_bytes()
            ).hexdigest(),
            full.ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
        )

    def test_smoke_units_and_cross_model_permutations_are_fixed(self):
        loader = DataLoader(str(ROOT / full.DATASET_PATH))
        loader.load_dataset()
        pairs = loader.get_question_pairs(
            str(ROOT / "data/pairs/country_pairs_smoke.json")
        )
        self.assertEqual(len(pairs), 2)
        prepared = {smoke.prepare_unit(loader, pair)["key"]: smoke.prepare_unit(loader, pair)
                    for pair in pairs}
        self.assertEqual(set(prepared), smoke.EXPECTED_UNITS)
        for key, expected in smoke.EXPECTED_DISPLAYED_OPTIONS.items():
            self.assertEqual(prepared[key]["displayed_options"], expected)

    def test_llama_serialization_contract_and_default_preamble(self):
        self.assertEqual(
            LLAMA31_DEFAULT_SYSTEM_MESSAGE,
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024",
        )
        messages = [
            {"role": "system", "content": LLAMA31_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": "Choose exactly one option label."},
        ]
        serialized = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + LLAMA31_DEFAULT_SYSTEM_MESSAGE
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n\n"
            + messages[1]["content"]
            + "<|eot_id|>"
            + LLAMA31_ANSWER_SUFFIX
        )
        verify_llama31_serialized_chat(serialized, messages)
        with self.assertRaises(RuntimeError):
            verify_llama31_serialized_chat(serialized + "A", messages)

    def test_raw_template_accepts_literal_or_jinja_bos_and_rejects_missing_bos(self):
        remainder = (
            f"{LLAMA31_HEADER_PREFIX} system assistant {LLAMA31_EOT} "
            "{% if add_generation_prompt %}"
        )
        for bos in (
            LLAMA31_BOS,
            "{{ bos_token }}",
            "{{- bos_token }}",
            "{{-bos_token-}}",
        ):
            with self.subTest(bos=bos):
                verify_llama31_chat_template(bos + remainder)

        with self.assertRaisesRegex(RuntimeError, "BOS literal or Jinja bos_token"):
            verify_llama31_chat_template(remainder)

    def test_cpp_helper_scores_complete_contextual_candidate_sequences(self):
        source = (ROOT / "src/gguf_score_helper.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "for (size_t offset = 0; offset < continuation_tokens.size(); ++offset)",
            source,
        )
        self.assertIn("sequence_log_probability += token_log_probability", source)
        self.assertIn("prompt tokenization is not a prefix", source)
        self.assertIn('{"raw_token_logits", raw_logits}', source)
        self.assertIn('{"prompt_bos_added", prompt_tokens.front() == llama_vocab_bos(vocab)}', source)

    def test_llama_outputs_use_explicit_metric_and_evidence_schema(self):
        paths = [
            ROOT / "scripts/run_genuine_llama_gguf_smoke.py",
            ROOT / "scripts/run_llama_gguf_full.py",
            ROOT / "scripts/analyze_llama_gguf_full.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("base2_jensen_shannon_distances", combined)
        self.assertIn("base2_jensen_shannon_divergences", combined)
        self.assertIn('"EO_raw"', combined)
        self.assertIn('"EO_normalized"', combined)
        self.assertEqual(full.EXPECTED_DIRECTED_UNIT_COUNT, 200)
        self.assertEqual(full.EXPECTED_ROW_COUNT, 800)

    def test_analysis_uses_question_clustered_bootstrap(self):
        self.assertEqual(analysis.EXPECTED_MODEL_REVISION, smoke.MODEL_REVISION)
        self.assertEqual(analysis.EXPECTED_MANIFEST_SHA256, full.PAIR_MANIFEST_SHA256)
        source = (ROOT / "scripts/analyze_llama_gguf_full.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("bootstrap_replicates: int = 10_000", source)
        self.assertIn("clustered_bootstrap(", source)
        bootstrap_source = (ROOT / "src/clustered_bootstrap.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("question_id", bootstrap_source)
        self.assertIn("condition rows are never sampled independently", source)


if __name__ == "__main__":
    unittest.main()
