from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from eval.adapters import EvalSample, resolve_lrs_image
from eval.metrics import build_prompt, extract_choice, extract_multi_choices, score_prediction, summarize_records
from eval.model import OPDVOpenAIGenerator
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
        generator = OPDVOpenAIGenerator(
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


if __name__ == "__main__":
    unittest.main()
