#!/usr/bin/env python3
"""Load Qwen3-VL through vLLM once and generate a tiny text response."""

from __future__ import annotations

import os


def main() -> None:
    model_path = os.environ["MODEL_PATH"]

    from vllm import LLM, SamplingParams

    print(f"Loading Qwen3-VL through vLLM: {model_path}", flush=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
    )
    result = llm.generate(
        ["Reply with exactly the word: ok"],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    print("QWEN3_VL_VLLM_SMOKE_OUTPUT=" + result[0].outputs[0].text, flush=True)
    print("Qwen3-VL vLLM smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
