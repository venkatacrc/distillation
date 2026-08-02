#!/usr/bin/env python
"""Capstone step 1: generate the full SFT training corpus from the
teacher - a mix of general instructions (Alpaca) and math reasoning
problems (GSM8K) - combining lab03's response-based recipe and lab06's
reasoning-distillation recipe into one unified data-generation pass."""
from __future__ import annotations

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt, load_alpaca, load_gsm8k  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    from vllm import LLM, SamplingParams

    data_cfg = cfg["data"]
    gen_cfg = cfg["generation"]

    alpaca_total = data_cfg["alpaca_train"] + data_cfg["alpaca_eval"]
    alpaca_ds = load_alpaca(n=alpaca_total, seed=cfg["seed"])
    alpaca_train = alpaca_ds.select(range(data_cfg["alpaca_train"]))
    alpaca_eval = alpaca_ds.select(range(data_cfg["alpaca_train"], alpaca_total))

    gsm8k_train = load_gsm8k("train", n=data_cfg["gsm8k_train"], seed=cfg["seed"])
    gsm8k_eval = load_gsm8k("test", n=data_cfg["gsm8k_eval"], seed=cfg["seed"])

    print(f"Loading teacher {cfg['teacher_model']} with vLLM...")
    llm = LLM(model=cfg["teacher_model"], dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=gen_cfg["max_new_tokens"], temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"]
    )

    def gen_split(name, alpaca_subset, gsm8k_subset):
        prompts, meta = [], []
        for ex in alpaca_subset:
            user_content = f"{ex['instruction']}\n\n{ex['input']}" if ex.get("input") else ex["instruction"]
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}], add_generation_prompt=True, tokenize=False
                )
            )
            meta.append({"instruction": user_content, "source": "alpaca"})
        for ex in gsm8k_subset:
            user_content = gsm8k_prompt(ex["question"])
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}], add_generation_prompt=True, tokenize=False
                )
            )
            meta.append({"instruction": user_content, "source": "gsm8k", "gold_answer": ex["answer"]})

        print(
            f"Generating {len(prompts)} teacher responses for split={name} "
            f"({len(alpaca_subset)} instructions + {len(gsm8k_subset)} math problems)..."
        )
        outputs = llm.generate(prompts, sampling_params)
        rows = [{**m, "response": o.outputs[0].text.strip()} for m, o in zip(meta, outputs)]

        out_path = os.path.join(LAB_DIR, f"results/teacher_data_{name}.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} rows to {out_path}")

    gen_split("train", alpaca_train, gsm8k_train)
    gen_split("eval", alpaca_eval, gsm8k_eval)


if __name__ == "__main__":
    main()
