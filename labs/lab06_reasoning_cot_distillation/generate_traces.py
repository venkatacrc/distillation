#!/usr/bin/env python
"""Lab06 step 1: rejection-sampling generation - draw K reasoning traces
per GSM8K training problem from an R1-lineage teacher, mirroring how
DeepSeek-R1's own ~800k-sample distillation set was built.

Reference: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability
in LLMs via Reinforcement Learning," 2025, Section 2.4.
"""
from __future__ import annotations

import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt, load_gsm8k  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    from vllm import LLM, SamplingParams

    gen_cfg = cfg["generation"]
    train_ds = load_gsm8k("train", n=cfg["train_problems"], seed=cfg["seed"])

    print(f"Loading teacher {cfg['teacher_model']} with vLLM (this is a 32B model - may take a few minutes)...")
    llm = LLM(model=cfg["teacher_model"], dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
    tokenizer = llm.get_tokenizer()

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": gsm8k_prompt(ex["question"])}], add_generation_prompt=True, tokenize=False
        )
        for ex in train_ds
    ]

    # `n=K` asks vLLM for K independent samples per prompt in a single
    # batched call - this *is* rejection sampling's generation step; the
    # "rejection" (keeping only correct+readable ones) happens in
    # filter_traces.py.
    sampling_params = SamplingParams(
        n=gen_cfg["k_samples_per_problem"],
        max_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )

    print(f"Generating {gen_cfg['k_samples_per_problem']} samples for each of {len(prompts)} GSM8K problems...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    rows = []
    for ex, out in zip(train_ds, outputs):
        for completion in out.outputs:
            rows.append({"question": ex["question"], "gold_answer": ex["answer"], "trace": completion.text})

    out_path = os.path.join(LAB_DIR, "results/raw_traces.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(
        f"Saved {len(rows)} raw traces ({len(train_ds)} problems x "
        f"{gen_cfg['k_samples_per_problem']} samples) to {out_path}  ({elapsed:.0f}s)"
    )

    json.dump(
        {"elapsed_seconds": elapsed, "n_problems": len(train_ds), "k_samples_per_problem": gen_cfg["k_samples_per_problem"]},
        open(os.path.join(LAB_DIR, "results/generation_timing.json"), "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
