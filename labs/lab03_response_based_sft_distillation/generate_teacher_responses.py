#!/usr/bin/env python
"""Lab03 step 1: batch-generate teacher completions for an Alpaca-style
instruction set with vLLM.

These sampled responses become the SFT target in train_student_sft.py -
this is "sequence-level" / response-based distillation: the student never
sees the teacher's logits (contrast with lab02/lab04), only its sampled
text, and is trained with plain next-token cross-entropy against it.
"""
from __future__ import annotations

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_alpaca  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def build_prompt(tokenizer, example):
    user_content = f"{example['instruction']}\n\n{example['input']}" if example.get("input") else example["instruction"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], add_generation_prompt=True, tokenize=False
    )
    return prompt, user_content


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    from vllm import LLM, SamplingParams

    total = cfg["train_prompts"] + cfg["eval_prompts"]
    full_ds = load_alpaca(n=total, seed=cfg["seed"])
    train_ds = full_ds.select(range(cfg["train_prompts"]))
    eval_ds = full_ds.select(range(cfg["train_prompts"], total))  # disjoint held-out slice

    print(f"Loading teacher {cfg['teacher_model']} with vLLM...")
    llm = LLM(model=cfg["teacher_model"], dtype="bfloat16", gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()

    gen_cfg = cfg["generation"]
    sampling_params = SamplingParams(
        max_tokens=gen_cfg["max_new_tokens"], temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"]
    )

    for split_name, ds in [("train", train_ds), ("eval", eval_ds)]:
        built = [build_prompt(tokenizer, ex) for ex in ds]
        prompts, user_contents = zip(*built)
        print(f"Generating {len(prompts)} teacher responses for split={split_name}...")
        outputs = llm.generate(list(prompts), sampling_params)
        rows = [
            {"instruction": uc, "response": out.outputs[0].text.strip()}
            for uc, out in zip(user_contents, outputs)
        ]
        out_path = os.path.join(LAB_DIR, f"results/teacher_responses_{split_name}.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        avg_len = sum(len(r["response"].split()) for r in rows) / len(rows)
        print(f"Saved {len(rows)} rows to {out_path} (avg {avg_len:.0f} words/response)")


if __name__ == "__main__":
    main()
