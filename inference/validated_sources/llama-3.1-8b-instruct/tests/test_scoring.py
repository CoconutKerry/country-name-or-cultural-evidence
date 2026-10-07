import math
from contextlib import nullcontext
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.model_runner import ModelRunner
from src.scoring import (
    map_label_probabilities,
    normalize_log_scores,
    option_labels,
    recover_canonical_distribution,
)


class OptionLabelScoringTests(unittest.TestCase):
    def test_shared_answer_prefixes_are_scored_by_option_label(self):
        """Long answers sharing a first word must retain distinct A/B scores."""
        class InjectedLabelRunner(ModelRunner):
            def __init__(self):
                super().__init__("test-model", "test://no-download")
                self.scored_labels = None

            def score_label_log_likelihoods(self, prompt, labels):
                self.scored_labels = list(labels)
                return [math.log(0.8), math.log(0.2)]

        displayed_options = [
            "Shared opening, but this is the first complete answer",
            "Shared opening, but this is the second complete answer",
        ]
        runner = InjectedLabelRunner()

        distribution = runner.predict_distribution(
            "Question with options. Answer:",
            displayed_options,
        )

        self.assertEqual(runner.scored_labels, ["A", "B"])
        self.assertEqual(set(distribution), set(displayed_options))
        self.assertAlmostEqual(distribution[displayed_options[0]], 0.8, places=12)
        self.assertAlmostEqual(distribution[displayed_options[1]], 0.2, places=12)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=12)

    def test_log_score_normalization_is_stable(self):
        probabilities = normalize_log_scores([1000.0, 999.0])

        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertAlmostEqual(probabilities[0], 0.7310585786, places=9)
        self.assertAlmostEqual(probabilities[1], 0.2689414214, places=9)

    def test_log_score_normalization_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            normalize_log_scores([])
        with self.assertRaises(ValueError):
            normalize_log_scores([0.0, float("nan")])
        with self.assertRaises(ValueError):
            normalize_log_scores([0.0, float("inf")])

    def test_option_permutation_recovers_semantic_distribution(self):
        canonical_options = ["Yes", "No", "DK"]
        displayed_options = ["DK", "Yes", "No"]
        label_probabilities = [0.1, 0.7, 0.2]

        displayed_distribution = map_label_probabilities(
            option_labels(len(displayed_options)),
            displayed_options,
            label_probabilities,
        )
        recovered = recover_canonical_distribution(
            displayed_distribution,
            canonical_options,
        )

        expected = {"Yes": 0.7, "No": 0.2, "DK": 0.1}
        self.assertEqual(set(recovered), set(canonical_options))
        self.assertEqual(list(recovered), canonical_options)
        for option, probability in expected.items():
            self.assertAlmostEqual(recovered[option], probability, places=12)
        self.assertAlmostEqual(sum(recovered.values()), 1.0, places=12)

    def test_production_loop_sums_all_tokens_of_each_label(self):
        class FakeTensor:
            def __init__(self, data):
                self.data = data

        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeLogProbs:
            def __init__(self, values):
                self.values = values

            def float(self):
                return self

            def __getitem__(self, token_id):
                return FakeScalar(self.values[token_id])

        class FakeLogits:
            def __init__(self, sequence, table):
                self.sequence = tuple(sequence)
                self.table = table

            def __getitem__(self, key):
                batch, position, vocab_slice = key
                self_test.assertEqual(batch, 0)
                self_test.assertEqual(vocab_slice, slice(None))
                return FakeLogProbs(self.table[(self.sequence, position)])

        class FakeModel:
            def __init__(self, table):
                self.table = table

            def __call__(self, *, input_ids, attention_mask, use_cache):
                self_test.assertFalse(use_cache)
                return SimpleNamespace(logits=FakeLogits(input_ids.data[0], self.table))

        class FakeTokenizer:
            def encode(self, text, *, add_special_tokens):
                return {
                    ("PROMPT", True): [1, 2],
                    ("PROMPT A", True): [1, 2, 10, 11],
                    ("PROMPT B", True): [1, 2, 10, 12],
                }[(text, add_special_tokens)]

        self_test = self
        table = {
            ((1, 2, 10, 11), 1): {10: math.log(0.5)},
            ((1, 2, 10, 11), 2): {11: math.log(0.8)},
            ((1, 2, 10, 12), 1): {10: math.log(0.5)},
            ((1, 2, 10, 12), 2): {12: math.log(0.2)},
        }
        fake_torch = SimpleNamespace(
            long=object(),
            inference_mode=nullcontext,
            tensor=lambda data, **kwargs: FakeTensor(data),
            ones_like=lambda tensor: FakeTensor([[1] * len(tensor.data[0])]),
            log_softmax=lambda vector, dim: vector,
        )
        runner = ModelRunner(
            "fake", "fake://model", use_chat_template=False
        )
        runner.tokenizer = FakeTokenizer()
        runner.model = FakeModel(table)
        runner.device = "fake-cpu"

        with patch.dict(sys.modules, {"torch": fake_torch}):
            scores = runner.score_label_log_likelihoods("PROMPT", ["A", "B"])

        self.assertAlmostEqual(scores[0], math.log(0.5 * 0.8), places=12)
        self.assertAlmostEqual(scores[1], math.log(0.5 * 0.2), places=12)
        self.assertNotAlmostEqual(scores[0], scores[1], places=12)
        metadata = runner.get_last_scoring_metadata()
        self.assertEqual(metadata["prompt_token_count"], 2)
        self.assertTrue(metadata["prompt_prefix_verified"])
        self.assertEqual(metadata["candidate_label_token_ids"], {
            "A": [10, 11], "B": [10, 12]
        })
        self.assertEqual(metadata["label_target_token_positions"], {
            "A": [2, 3], "B": [2, 3]
        })
        self.assertEqual(metadata["label_predictive_logit_positions"], {
            "A": [1, 2], "B": [1, 2]
        })

    def test_chat_template_and_label_separator_are_model_specific(self):
        class FakeTokenizer:
            chat_template = "fake-template"

            def __init__(self):
                self.call = None

            def apply_chat_template(self, messages, **kwargs):
                self.call = (messages, kwargs)
                return "<assistant>\n"

        tokenizer = FakeTokenizer()
        runner = ModelRunner("fake", "fake://model", use_chat_template=True)
        runner.tokenizer = tokenizer

        rendered = runner.render_prompt("QUESTION")

        self.assertEqual(rendered, "<assistant>\n")
        self.assertEqual(tokenizer.call[0], [{"role": "user", "content": "QUESTION"}])
        self.assertEqual(
            tokenizer.call[1],
            {"tokenize": False, "add_generation_prompt": True},
        )
        self.assertEqual(runner._label_continuation("A", rendered), "A")
        self.assertEqual(runner._label_continuation("A", "[/INST]"), " A")

    def test_hub_verified_revision_does_not_require_private_tokenizer_commit_field(self):
        class FakeTokenizer:
            init_kwargs = {}
            chat_template = "template"
            pad_token = None
            eos_token = "<eos>"

        class FakeModel:
            config = SimpleNamespace(_commit_hash=None)
            dtype = "float16"
            hf_device_map = {"": 0}

            def eval(self):
                return self

            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cuda:0"))

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return FakeTokenizer()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return FakeModel()

        class FakeHfApi:
            def __init__(self, token=None):
                pass

            def model_info(self, hub_id, revision, token=None):
                return SimpleNamespace(sha=revision)

        fake_torch = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda="test-cuda"),
            cuda=SimpleNamespace(is_available=lambda: False),
            device=lambda name: name,
        )
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
        )
        fake_hub = SimpleNamespace(HfApi=FakeHfApi)
        runner = ModelRunner(
            "fake",
            "fake/model",
            revision="abc123",
            tokenizer_revision="abc123",
            trust_remote_code=False,
        )

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
                "huggingface_hub": fake_hub,
            },
        ):
            runner.load_model()

        runtime = runner.get_runtime_metadata()
        self.assertEqual(runtime["resolved_model_revision"], "abc123")
        self.assertEqual(runtime["resolved_tokenizer_revision"], "abc123")
        self.assertIsNone(runtime["transformers_model_commit_metadata"])
        self.assertIsNone(runtime["transformers_tokenizer_commit_metadata"])


if __name__ == "__main__":
    unittest.main()
