#!/usr/bin/env python
"""Capstone: full evaluation suite - GSM8K pass@1, MMLU-lite, IFEval-lite,
and inference latency/throughput/memory - run once for each of the
teacher, the non-distilled baseline student, and the final distilled
student, so report.py can produce a single side-by-side comparison.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt, load_gsm8k, load_mmlu_subset  # noqa: E402
from common.eval_harness import score_gsm8k, score_mmlu_lite  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _ifeval_lite import score_ifeval_lite  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def vllm_generate(model_path, prompts, max_new_tokens=512, temperature=0.0, top_p=1.0, gpu_mem=0.6):
    from vllm import LLM, SamplingParams

    torch.cuda.reset_peak_memory_stats()
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_mem, max_model_len=4096, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    t0 = time.time()
    outputs = llm.generate(formatted, sampling_params)
    elapsed = time.time() - t0
    texts = [o.outputs[0].text for o in outputs]
    n_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts, {
        "elapsed_s": elapsed,
        "throughput_tok_per_s": n_out_tokens / elapsed if elapsed > 0 else float("nan"),
        "peak_mem_gb": peak_mem_gb,
    }


def evaluate_model(name: str, model_path: str, cfg: dict) -> dict:
    print(f"\n{'=' * 70}\nEvaluating {name} ({model_path})\n{'=' * 70}")
    results = {}

    gsm_ds = load_gsm8k("test", n=cfg["data"]["gsm8k_eval"], seed=cfg["seed"])
    prompts = [gsm8k_prompt(ex["question"]) for ex in gsm_ds]
    gold = [ex["answer"] for ex in gsm_ds]
    preds, _ = vllm_generate(model_path, prompts, max_new_tokens=512)
    gsm8k_score = score_gsm8k(preds, gold)
    results["gsm8k_pass_at_1"] = gsm8k_score["accuracy"]
    print(f"  GSM8K pass@1: {gsm8k_score['accuracy']:.1%}")

    def gen_fn(prompt_list):
        texts, _ = vllm_generate(model_path, prompt_list, max_new_tokens=256)
        return texts

    ifeval_acc, ifeval_results = score_ifeval_lite(gen_fn)
    results["ifeval_lite_accuracy"] = ifeval_acc
    print(f"  IFEval-lite: {ifeval_acc:.1%} ({sum(r['passed'] for r in ifeval_results)}/{len(ifeval_results)} constraints satisfied)")

    latency_prompts = ["Tell me an interesting fact about the ocean."] * cfg["eval"]["latency_n_prompts"]
    _, perf_stats = vllm_generate(model_path, latency_prompts, max_new_tokens=cfg["eval"]["latency_max_new_tokens"])
    results.update(perf_stats)
    print(f"  Throughput: {perf_stats['throughput_tok_per_s']:.0f} tok/s, peak mem: {perf_stats['peak_mem_gb']:.1f}GB")

    # MMLU-lite uses loglikelihood scoring, which needs direct model access
    # (not generation-based), so it's evaluated with plain transformers.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()
    mmlu_ds = load_mmlu_subset(cfg["eval"]["mmlu_subjects"], n_per_subject=cfg["eval"]["mmlu_n_per_subject"])
    mmlu_score = score_mmlu_lite(model, tokenizer, mmlu_ds)
    results["mmlu_lite_accuracy"] = mmlu_score["accuracy"]
    print(f"  MMLU-lite: {mmlu_score['accuracy']:.1%}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    candidates = {
        "teacher": cfg["teacher_model"],
        "baseline_student": cfg["student_base_model"],
    }
    sft_path = os.path.join(LAB_DIR, "results/sft_student")
    gkd_path = os.path.join(LAB_DIR, "results/gkd_student")
    if os.path.exists(sft_path):
        candidates["sft_student"] = sft_path
    if os.path.exists(gkd_path):
        candidates["distilled_student"] = gkd_path

    all_results = {}
    for name, path in candidates.items():
        all_results[name] = evaluate_model(name, path, cfg)

    out_path = os.path.join(LAB_DIR, "results/eval_suite_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
