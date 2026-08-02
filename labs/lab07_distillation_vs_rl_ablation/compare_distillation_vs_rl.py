#!/usr/bin/env python
"""Lab07 step 2: compare DeepSeek-R1-style SFT distillation (lab06)
against RL-from-scratch (this lab's GRPO run) on the same held-out GSM8K
test set, reporting both accuracy and approximate compute cost.

Reproduces the qualitative conclusion of DeepSeek-R1's own Table 6
ablation at small scale: distilling reasoning ability from a strong
teacher beats training the small model with RL directly, for a fraction
of the compute.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt, load_gsm8k  # noqa: E402
from common.eval_harness import score_gsm8k  # noqa: E402
from common.plotting import bar_plot, scatter_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)
LAB06_RESULTS = os.path.join(LAB_DIR, "..", "lab06_reasoning_cot_distillation", "results")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def vllm_generate(model_path, prompts, max_new_tokens=1024, temperature=0.0, top_p=1.0):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.7, max_model_len=4096, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    outputs = llm.generate(formatted, sampling_params)
    texts = [o.outputs[0].text for o in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    eval_ds = load_gsm8k("test", n=cfg["eval_problems"], seed=cfg["seed"])
    prompts = [gsm8k_prompt(ex["question"]) for ex in eval_ds]
    gold = [ex["answer"] for ex in eval_ds]

    grpo_path = os.path.join(LAB_DIR, "results/grpo_student")
    if not os.path.exists(grpo_path):
        raise FileNotFoundError(f"{grpo_path} not found - run train_grpo_baseline.py first.")
    cot_path = os.path.join(LAB06_RESULTS, "cot_student")
    if not os.path.exists(cot_path):
        raise FileNotFoundError(f"{cot_path} not found - run lab06 first (its distilled student is the comparison point).")

    print("Evaluating GRPO (RL-from-scratch) student...")
    grpo_preds = vllm_generate(grpo_path, prompts)
    grpo_score = score_gsm8k(grpo_preds, gold)
    print(f"  GRPO pass@1: {grpo_score['accuracy']:.1%}")

    print("Evaluating lab06 SFT-distilled (DeepSeek-R1-style) student...")
    distill_preds = vllm_generate(cot_path, prompts, max_new_tokens=1024)
    distill_score = score_gsm8k(distill_preds, gold)
    print(f"  Distillation pass@1: {distill_score['accuracy']:.1%}")

    grpo_time = load_json(os.path.join(LAB_DIR, "results/grpo_train_summary.json"))["elapsed_seconds"]
    distill_time = 0.0
    for fname in ["generation_timing.json", "sft_timing.json"]:
        path = os.path.join(LAB06_RESULTS, fname)
        if os.path.exists(path):
            distill_time += load_json(path)["elapsed_seconds"]

    bar_plot(
        ["RL from scratch\n(GRPO)", "Distillation\n(lab06, SFT on R1 traces)"],
        [grpo_score["accuracy"], distill_score["accuracy"]],
        ylabel="GSM8K pass@1",
        title="Distillation vs. RL-from-scratch on the same small model",
        out_path=os.path.join(LAB_DIR, "results/distillation_vs_rl_accuracy.png"),
    )

    if distill_time > 0:
        scatter_plot(
            [grpo_time / 3600, distill_time / 3600],
            [grpo_score["accuracy"], distill_score["accuracy"]],
            xlabel="approx. compute (GPU-hours, this run)",
            ylabel="GSM8K pass@1",
            title="Accuracy vs. compute: distillation vs. RL-from-scratch",
            out_path=os.path.join(LAB_DIR, "results/accuracy_vs_compute.png"),
            labels=["GRPO (RL)", "Distillation"],
        )
    else:
        print("(lab06 timing files not found - skipping accuracy-vs-compute plot; re-run lab06 to generate them)")

    json.dump(
        {
            "grpo_pass_at_1": grpo_score["accuracy"],
            "distillation_pass_at_1": distill_score["accuracy"],
            "grpo_elapsed_seconds": grpo_time,
            "distillation_elapsed_seconds": distill_time or None,
            "n_eval": len(prompts),
        },
        open(os.path.join(LAB_DIR, "results/comparison.json"), "w"),
        indent=2,
    )
    print("Saved results/distillation_vs_rl_accuracy.png and results/comparison.json")


if __name__ == "__main__":
    main()
