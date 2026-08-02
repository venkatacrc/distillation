#!/usr/bin/env python
"""Lab06 step 3: plain SFT (no RL) of a small base model on the filtered
CoT traces.

This matches DeepSeek-R1's own finding for its distilled model family:
the small models are trained with ordinary supervised fine-tuning on
curated reasoning traces, *not* with reinforcement learning - RL is only
used to train the big teacher that generates those traces in the first
place (Section 2.4: "we directly fine-tuned open-source models ... using
the 800k samples curated with DeepSeek-R1 ... We did not apply an RL stage
for these distilled models"). Lab07 stress-tests this design choice.
"""
from __future__ import annotations

import json
import os
import sys
import time

import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    filtered_path = os.path.join(LAB_DIR, "results/filtered_traces.jsonl")
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(f"{filtered_path} not found - run filter_traces.py first.")
    rows = load_jsonl(filtered_path)

    train_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": gsm8k_prompt(r["question"])},
                    {"role": "assistant", "content": r["trace"]},
                ]
            }
            for r in rows
        ]
    )
    print(f"Training on {len(train_dataset)} filtered chain-of-thought traces")

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["student_base_model"], dtype="bfloat16")

    sft_cfg = cfg["sft"]
    output_dir = os.path.join(LAB_DIR, "results/cot_student")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["learning_rate"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        max_length=sft_cfg["max_length"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset, processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved CoT-distilled student to {output_dir}  ({elapsed:.0f}s)")

    json.dump(
        {"elapsed_seconds": elapsed, "n_train_examples": len(train_dataset)},
        open(os.path.join(LAB_DIR, "results/sft_timing.json"), "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
