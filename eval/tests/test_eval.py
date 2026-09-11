from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from eval.adapters import EvalSample, resolve_lrs_image
from eval.lrs_semantic import (
    annotate_lrs_semantic_records,
    canonicalize_lrs_answer,
    is_protected_answer,
    write_jsonl_atomic,
)
from eval.rescore_lrs_tolerant import load_lrs_records, rescore_records
from eval.metrics import build_prompt, extract_choice, extract_multi_choices, score_prediction, summarize_records
from eval.model import OpenAICompatibleGenerator
from eval.run import selected_datasets


class EvalTests(unittest.TestCase):
    def make_sample(self, dataset: str, **kwargs) -> EvalSample:
        defaults = {
            "dataset": dataset,
            "sample_id": "sample",
            "image_ref": Path("unused.png"),
            "image_display_path": "unused.png",
            "question": "Question?",
            "ground_truth": "A",
            "choices": ["(A) alpha", "(B) beta", "(C) gamma", "(D) delta", "(E) none"],
            "metadata": {"category": "category"},
        }
        defaults.update(kwargs)
        return EvalSample(**defaults)

    def test_dataset_selection_and_lrs_flattened_path(self) -> None:
        self.assertEqual(selected_datasets(["lrs-vqa,xlrs-bench"]), ("lrs-vqa", "xlrs-bench"))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "image").mkdir()
            image_path = root / "image" / "123.tif"
            Image.new("RGB", (2, 2)).save(image_path)
            self.assertEqual(resolve_lrs_image(root, "LRS_VQA/image/123.tif"), image_path)

    def test_scoring_for_each_answer_format(self) -> None:
        self.assertTrue(score_prediction(self.make_sample("lrs-vqa", ground_truth="Yes"), " yes. ")["correct"])
        self.assertTrue(score_prediction(self.make_sample("mme-realworld-rs", ground_truth="E"), "(E)")["correct"])
        xlrs = self.make_sample(
            "xlrs-bench",
            ground_truth="BCD",
            metadata={"category": "Land use classification/Overall Land use classification", "is_multi_choice": True},
        )
        scored = score_prediction(xlrs, "The answers are B, C, and D.")
        self.assertEqual(scored["parsed_answer"], "B,C,D")
        self.assertTrue(scored["correct"])
        self.assertEqual(extract_multi_choices("BCD"), {"B", "C", "D"})
        self.assertEqual(extract_choice("Choice (E)"), "E")
        self.assertIn("more than one", build_prompt(xlrs))

    def test_openai_adapter_encodes_and_normalizes_without_openai_dependency(self) -> None:
        generator = OpenAICompatibleGenerator(
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            model_id="model",
            max_tokens=10,
            max_retries=1,
            request_timeout=1,
            enable_thinking=None,
            max_pixels=4,
            image_format="png",
            jpeg_quality=95,
        )
        uri = generator.image_to_data_uri(Image.new("RGB", (10, 10), color="red"))
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertTrue(base64.b64decode(uri.split(",", 1)[1]))
        self.assertEqual(generator.normalize_model_answer("<think>x</think><answer>B</answer>"), "B")

    def test_summary_accounts_for_errors(self) -> None:
        summary = summarize_records([
            {"dataset": "lrs-vqa", "status": "ok", "correct": True, "metrics": {}, "metadata": {}, "elapsed_sec": 2.0},
            {"dataset": "lrs-vqa", "status": "error", "correct": False, "metrics": {}, "metadata": {}, "elapsed_sec": 5.0},
        ])
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["total_inference_sec"], 2.0)

    def test_lrs_tolerant_semantic_scoring_keeps_strict_results(self) -> None:
        class FakeSimilarity:
            def __init__(self) -> None:
                self.pairs: list[tuple[str, str]] = []

            def score_pairs(self, pairs):
                self.pairs = list(pairs)
                return [0.84]

        records = [
            {"dataset": "lrs-vqa", "status": "ok", "correct": False, "prediction": "rectangle", "ground_truth": "rectangular", "metrics": {}, "metadata": {}, "elapsed_sec": 1.0},
            {"dataset": "lrs-vqa", "status": "ok", "correct": False, "prediction": "no", "ground_truth": "yes", "metrics": {}, "metadata": {}, "elapsed_sec": 1.0},
            {"dataset": "lrs-vqa", "status": "ok", "correct": False, "prediction": "2", "ground_truth": "3", "metrics": {}, "metadata": {}, "elapsed_sec": 1.0},
            {"dataset": "lrs-vqa", "status": "ok", "correct": False, "prediction": "green", "ground_truth": "blue", "metrics": {}, "metadata": {}, "elapsed_sec": 1.0},
            {"dataset": "lrs-vqa", "status": "ok", "correct": True, "prediction": "yes", "ground_truth": "yes", "metrics": {}, "metadata": {}, "elapsed_sec": 1.0},
            {"dataset": "lrs-vqa", "status": "error", "correct": False, "prediction": "", "ground_truth": "urban", "metrics": {}, "metadata": {}, "elapsed_sec": 0.0},
            {"dataset": "mme-realworld-rs", "status": "ok", "correct": False, "metrics": {}, "metadata": {}, "elapsed_sec": 1.0},
        ]
        scorer = FakeSimilarity()
        annotate_lrs_semantic_records(records, scorer, threshold=0.85)

        self.assertEqual(scorer.pairs, [("green", "blue")])
        self.assertFalse(records[0]["correct"])
        self.assertTrue(records[0]["tolerant_correct"])
        self.assertEqual(records[0]["tolerant_match_source"], "canonical_alias")
        self.assertFalse(records[0]["semantic_scored"])
        self.assertEqual(records[1]["tolerant_match_source"], "strict_guard")
        self.assertEqual(records[2]["tolerant_match_source"], "strict_guard")
        self.assertFalse(records[3]["tolerant_correct"])
        self.assertEqual(records[3]["tolerant_match_source"], "not_matched")
        self.assertTrue(records[4]["tolerant_correct"])
        self.assertEqual(records[5]["tolerant_match_source"], "error")
        self.assertNotIn("tolerant_correct", records[6])
        self.assertTrue(is_protected_answer("second"))
        self.assertFalse(is_protected_answer("stationary"))

        summary = summarize_records(records)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["tolerant_correct"], 2)
        self.assertEqual(summary["tolerant_rescued"], 1)
        self.assertEqual(summary["canonical_alias_rescued"], 1)
        self.assertEqual(summary["semantic_rescued"], 0)
        self.assertEqual(summary["semantic_scored"], 1)
        self.assertAlmostEqual(summary["tolerant_accuracy"], 2 / 6)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl_atomic(output_path, records)
            persisted = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(persisted[0]["tolerant_match_source"], "canonical_alias")


if __name__ == "__main__":
    unittest.main()
