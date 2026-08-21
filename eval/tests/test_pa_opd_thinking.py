from __future__ import annotations

from pathlib import Path
import unittest

from eval.adapters import EvalSample
from eval.pa_opd_thinking import (
    answer_candidate,
    build_pa_opd_prompt,
    build_pa_opd_trace,
    completion_after_prefilled_think,
    is_valid_pa_opd_format,
)


class PAOPDThinkingTests(unittest.TestCase):
    def make_sample(self) -> EvalSample:
        return EvalSample(
            dataset="mme-realworld-rs",
            sample_id="sample",
            image_ref=Path("unused.png"),
            image_display_path="unused.png",
            question="What color is the object?",
            ground_truth="A",
            choices=["A. yellow", "B. red", "C. blue", "D. brown"],
            metadata={"category": "category"},
        )

    def test_prompt_and_rollout_fields_match_pa_opd_prefix_protocol(self) -> None:
        prompt = build_pa_opd_prompt(self.make_sample())
        self.assertEqual(
            prompt,
            "What color is the object?\n\nA. yellow\nB. red\nC. blue\nD. brown"
            "\n\nAnswer with the option's letter from the given choices.\n\n"
            "Continue the already-open <think> block with reasoning, then emit "
            "</think><answer>the option label only</answer>.",
        )
        trace = build_pa_opd_trace(prompt, "It is yellow.\n</think><answer>A</answer>")
        self.assertTrue(trace.format_valid)
        self.assertEqual(trace.answer, "A")
        self.assertEqual(trace.reasoning, "It is yellow.\n")
        self.assertTrue(trace.input.endswith("assistant\n<think>\n"))
        self.assertEqual(trace.canonical_response, "<think>It is yellow.\n</think><answer>A</answer>")

    def test_parser_removes_only_a_server_duplicated_opening_tag(self) -> None:
        response = "<think>reasoning</think><answer>B</answer>"
        self.assertEqual(completion_after_prefilled_think(response), "reasoning</think><answer>B</answer>")
        trace = build_pa_opd_trace("Question?", response)
        self.assertTrue(trace.format_valid)
        self.assertEqual(answer_candidate(trace), "B")
        self.assertFalse(is_valid_pa_opd_format("reason</think><answer>A</answer></think>"))

    def test_invalid_format_still_retains_a_best_effort_answer_for_benchmark_scoring(self) -> None:
        trace = build_pa_opd_trace("Question?", "reasoning</think><answer>C</answer></think>")
        self.assertFalse(trace.format_valid)
        self.assertEqual(trace.answer, None)
        self.assertEqual(answer_candidate(trace), "C")


if __name__ == "__main__":
    unittest.main()
