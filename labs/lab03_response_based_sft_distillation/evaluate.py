#!/usr/bin/env python
"""Lab03 step 3: compare the SFT-distilled student against (a) the same
base model with no distillation and (b) the teacher itself, on a held-out
set of instructions, using a pairwise LLM-judge win-rate.

Caveat: this reuses the teacher as judge for simplicity, which is known to
be biased toward the teacher's own style/verbosity ("self-preference
bias" - see Zheng et al. 2023, "Judging LLM-as-a-Judge"). Treat these
win-rates as directionally informative, not an exact/fair benchmark. Swap
in a different judge model in config.yaml if you have one available.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.eval_harness import llm_judge_winrate  # noqa: E402
from common.plotting import bar_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def vllm_generate(model_path, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    outputs = llm.generate(formatted, sampling_params)
    texts = [o.outputs[0].text.strip() for o in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    eval_path = os.path.join(LAB_DIR, "results/teacher_responses_eval.jsonl")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"{eval_path} not found - run generate_teacher_responses.py first.")
    eval_rows = load_jsonl(eval_path)
    prompts = [r["instruction"] for r in eval_rows]
    teacher_responses = [r["response"] for r in eval_rows]

    gen_cfg = cfg["generation"]

    print(f"Generating baseline (non-distilled) student responses from {cfg['student_base_model']}...")
    baseline_responses = vllm_generate(
        cfg["student_base_model"], prompts, gen_cfg["max_new_tokens"], gen_cfg["temperature"], gen_cfg["top_p"]
    )

    sft_path = os.path.join(LAB_DIR, "results/sft_student")
    if not os.path.exists(sft_path):
        raise FileNotFoundError(f"{sft_path} not found - run train_student_sft.py first.")
    print(f"Generating SFT-distilled student responses from {sft_path}...")
    sft_responses = vllm_generate(sft_path, prompts, gen_cfg["max_new_tokens"], gen_cfg["temperature"], gen_cfg["top_p"])

    print(f"Loading judge {cfg['judge']['model']}...")
    from vllm import LLM, SamplingParams

    judge_llm = LLM(model=cfg["judge"]["model"], dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    judge_tokenizer = judge_llm.get_tokenizer()
    judge_sampling = SamplingParams(max_tokens=cfg["judge"]["max_new_tokens"], temperature=0.0)

    def judge_generate_fn(judge_prompts):
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
            )
            for p in judge_prompts
        ]
        outputs = judge_llm.generate(formatted, judge_sampling)
        return [o.outputs[0].text.strip() for o in outputs]

    print("Judging: SFT-distilled student vs. baseline (non-distilled) student...")
    result_vs_baseline = llm_judge_winrate(judge_generate_fn, prompts, sft_responses, baseline_responses)
    print(
        f"  SFT student win-rate vs. baseline: {result_vs_baseline.win_rate_a:.1%} "
        f"({result_vs_baseline.n_valid}/{result_vs_baseline.n_total} decisive judgments)"
    )

    print("Judging: SFT-distilled student vs. teacher...")
    result_vs_teacher = llm_judge_winrate(judge_generate_fn, prompts, sft_responses, teacher_responses)
    print(
        f"  SFT student win-rate vs. teacher:  {result_vs_teacher.win_rate_a:.1%} "
        f"({result_vs_teacher.n_valid}/{result_vs_teacher.n_total} decisive judgments)"
    )

    bar_plot(
        ["SFT student\nvs. baseline student", "SFT student\nvs. teacher"],
        [result_vs_baseline.win_rate_a, result_vs_teacher.win_rate_a],
        ylabel="SFT-distilled student win-rate",
        title="Response-based SFT distillation: win-rate vs. baseline and teacher",
        out_path=os.path.join(LAB_DIR, "results/win_rates.png"),
    )

    json.dump(
        {
            "vs_baseline_win_rate": result_vs_baseline.win_rate_a,
            "vs_baseline_n_valid": result_vs_baseline.n_valid,
            "vs_teacher_win_rate": result_vs_teacher.win_rate_a,
            "vs_teacher_n_valid": result_vs_teacher.n_valid,
            "n_eval_prompts": len(prompts),
        },
        open(os.path.join(LAB_DIR, "results/eval_results.json"), "w"),
        indent=2,
    )
    print("Saved results/eval_results.json and results/win_rates.png")


if __name__ == "__main__":
    main()
