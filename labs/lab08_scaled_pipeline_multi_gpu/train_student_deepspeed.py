#!/usr/bin/env python
"""Lab08 step 2: SFT the student at a larger scale (Qwen2.5-3B) on the
teacher's API-served responses, using DeepSpeed ZeRO-3 to shard optimizer
state, gradients, and parameters across the GPUs *not* running the teacher
server.

Launch (assuming serve_teacher_vllm.sh is using GPUs 0-1, leaving 2-7
free):

    deepspeed --num_gpus=6 --include localhost:2,3,4,5,6,7 \\
        train_student_deepspeed.py
"""
from __future__ import annotations

import json
import os
import sys

import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

LAB_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    train_path = os.path.join(LAB_DIR, "results/teacher_responses_train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found - run generate_teacher_responses_api.py first.")
    rows = load_jsonl(train_path)
    train_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": r["instruction"]},
                    {"role": "assistant", "content": r["response"]},
                ]
            }
            for r in rows
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["student_base_model"], dtype="bfloat16")

    sft_cfg = cfg["sft"]
    output_dir = os.path.join(LAB_DIR, "results/sft_student_scaled")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["learning_rate"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        max_length=sft_cfg["max_length"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        bf16=True,
        deepspeed=os.path.join(LAB_DIR, "ds_config_zero3.json"),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        print(f"Saved scaled SFT student to {output_dir}")


if __name__ == "__main__":
    main()
