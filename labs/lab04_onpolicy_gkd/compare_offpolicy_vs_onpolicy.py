#!/usr/bin/env python
"""Lab04 step 2: compare the on-policy GKD sweep against the lab03
off-policy SFT baseline using an LLM-judge win-rate, and check whether
`beta` (forward vs. reverse KL, see lab02) affects response length/style
the way the theory predicts (reverse-KL-leaning runs should be terser and
more "committed" to one answer; forward-KL-leaning runs more hedged)."""
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
LAB03_RESULTS = os.path.join(LAB_DIR, "..", "lab03_response_based_sft_distillation", "results")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def vllm_generate(model_path, prompts, max_new_tokens=128, temperature=0.7, top_p=0.9):
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
    manifest_path = os.path.join(LAB_DIR, "results/sweep_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} not found - run train_gkd.py first.")
    manifest = json.load(open(manifest_path))

    eval_rows = load_jsonl(os.path.join(LAB03_RESULTS, "teacher_responses_eval.jsonl"))
    prompts = [r["instruction"] for r in eval_rows][: cfg["eval_prompts"]]

    sft_path = os.path.join(LAB03_RESULTS, "sft_student")
    if not os.path.exists(sft_path):
        raise FileNotFoundError(f"{sft_path} not found - run lab03 first (used here as the off-policy baseline).")
    print("Generating baseline (lab03 off-policy SFT student) responses...")
    baseline_responses = vllm_generate(sft_path, prompts)

    print(f"Loading judge {cfg['teacher_model']}...")
    from vllm import LLM, SamplingParams

    judge_llm = LLM(model=cfg["teacher_model"], dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    judge_tokenizer = judge_llm.get_tokenizer()
    judge_sampling = SamplingParams(max_tokens=8, temperature=0.0)

    def judge_generate_fn(judge_prompts):
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
            )
            for p in judge_prompts
        ]
        outputs = judge_llm.generate(formatted, judge_sampling)
        return [o.outputs[0].text.strip() for o in outputs]

    labels, win_rates, avg_lengths = [], [], []
    for run in manifest:
        label = f"lmbda={run['lmbda']}\nbeta={run['beta']}"
        print(f"\nGenerating responses for {label.replace(chr(10), ' ')}...")
        responses = vllm_generate(run["output_dir"], prompts)
        avg_lengths.append(sum(len(r.split()) for r in responses) / len(responses))

        result = llm_judge_winrate(judge_generate_fn, prompts, responses, baseline_responses)
        print(
            f"  win-rate vs. lab03 off-policy SFT baseline: {result.win_rate_a:.1%} "
            f"({result.n_valid}/{result.n_total} decisive)"
        )
        labels.append(label)
        win_rates.append(result.win_rate_a)

    bar_plot(
        labels,
        win_rates,
        ylabel="win-rate vs. lab03 off-policy SFT baseline",
        title="On-policy GKD sweep vs. off-policy SFT baseline",
        out_path=os.path.join(LAB_DIR, "results/gkd_vs_offpolicy_winrate.png"),
    )
    bar_plot(
        labels,
        avg_lengths,
        ylabel="avg response length (words)",
        title="Response length by (lmbda, beta)",
        out_path=os.path.join(LAB_DIR, "results/gkd_response_length.png"),
    )

    json.dump(
        [
            {"label": l.replace("\n", " "), "win_rate_vs_offpolicy": w, "avg_length_words": a}
            for l, w, a in zip(labels, win_rates, avg_lengths)
        ],
        open(os.path.join(LAB_DIR, "results/comparison.json"), "w"),
        indent=2,
    )
    print("\nSaved results/gkd_vs_offpolicy_winrate.png, results/gkd_response_length.png, results/comparison.json")


if __name__ == "__main__":
    main()
