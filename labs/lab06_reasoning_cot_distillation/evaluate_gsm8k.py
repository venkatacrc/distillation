#!/usr/bin/env python
"""Lab06 step 4: measure GSM8K pass@1 before vs. after CoT distillation,
and show why rejection sampling (drawing K samples per problem) matters by
plotting, retrospectively, what fraction of training problems would have
had >=1 correct trace as a function of K."""
from __future__ import annotations

import gc
import json
import os
import sys
from collections import defaultdict

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt, load_gsm8k  # noqa: E402
from common.eval_harness import gsm8k_correct, score_gsm8k  # noqa: E402
from common.plotting import bar_plot, line_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


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


def rejection_sampling_coverage(cfg):
    """Reuses the already-generated raw_traces.jsonl to show, retrospectively,
    what fraction of problems would have had >=1 correct trace if we'd only
    drawn the first K of the samples per problem."""
    raw_path = os.path.join(LAB_DIR, "results/raw_traces.jsonl")
    if not os.path.exists(raw_path):
        print("Skipping rejection-sampling coverage plot: results/raw_traces.jsonl not found.")
        return
    rows = load_jsonl(raw_path)
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["question"]].append(r)

    max_k = cfg["generation"]["k_samples_per_problem"]
    ks = list(range(1, max_k + 1))
    coverage = []
    for k in ks:
        n_covered = sum(
            1 for traces in by_problem.values() if any(gsm8k_correct(t["trace"], t["gold_answer"]) for t in traces[:k])
        )
        coverage.append(n_covered / len(by_problem))

    line_plot(
        ks,
        {"fraction of problems with >=1 correct trace": coverage},
        xlabel="K (samples drawn per problem)",
        ylabel="coverage",
        title="Why rejection sampling matters: training-set coverage vs. K",
        out_path=os.path.join(LAB_DIR, "results/rejection_sampling_coverage.png"),
    )
    for k, c in zip(ks, coverage):
        print(f"  K={k}: {c:.1%} of problems have >=1 correct trace")
    print("Saved results/rejection_sampling_coverage.png")


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    eval_ds = load_gsm8k("test", n=cfg["eval_problems"], seed=cfg["seed"])
    prompts = [gsm8k_prompt(ex["question"]) for ex in eval_ds]
    gold = [ex["answer"] for ex in eval_ds]

    print(f"Evaluating baseline (non-distilled) {cfg['student_base_model']} on {len(prompts)} held-out GSM8K problems...")
    baseline_preds = vllm_generate(cfg["student_base_model"], prompts)
    baseline_score = score_gsm8k(baseline_preds, gold)
    print(f"  baseline pass@1: {baseline_score['accuracy']:.1%}")

    cot_path = os.path.join(LAB_DIR, "results/cot_student")
    if not os.path.exists(cot_path):
        raise FileNotFoundError(f"{cot_path} not found - run train_student_cot_sft.py first.")
    print(f"Evaluating CoT-distilled student on {len(prompts)} held-out GSM8K problems...")
    cot_preds = vllm_generate(cot_path, prompts, max_new_tokens=1024)
    cot_score = score_gsm8k(cot_preds, gold)
    print(f"  CoT-distilled pass@1: {cot_score['accuracy']:.1%}")

    bar_plot(
        [f"Baseline\n({cfg['student_base_model'].split('/')[-1]})", "CoT-distilled\n(this lab)"],
        [baseline_score["accuracy"], cot_score["accuracy"]],
        ylabel="GSM8K pass@1",
        title="Reasoning distillation: GSM8K accuracy before vs. after",
        out_path=os.path.join(LAB_DIR, "results/gsm8k_before_after.png"),
    )

    json.dump(
        {
            "baseline_pass_at_1": baseline_score["accuracy"],
            "cot_distilled_pass_at_1": cot_score["accuracy"],
            "n_eval": len(prompts),
        },
        open(os.path.join(LAB_DIR, "results/gsm8k_eval_results.json"), "w"),
        indent=2,
    )
    print("Saved results/gsm8k_before_after.png and results/gsm8k_eval_results.json")

    rejection_sampling_coverage(cfg)


if __name__ == "__main__":
    main()
