#!/usr/bin/env python
"""Capstone step 2: SFT the student on the full teacher-generated corpus
(general instructions + math reasoning) from step 1 - the offline,
response-based distillation stage (lab03/lab06's recipe)."""
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
    train_path = os.path.join(LAB_DIR, "results/teacher_data_train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found - run 01_generate_teacher_data.py first.")
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
    n_math = sum(1 for r in rows if r["source"] == "gsm8k")
    n_general = sum(1 for r in rows if r["source"] == "alpaca")
    print(f"Training on {len(train_dataset)} teacher-generated examples ({n_math} math, {n_general} general instructions)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["student_base_model"], dtype="bfloat16")

    sft_cfg = cfg["sft"]
    output_dir = os.path.join(LAB_DIR, "results/sft_student")
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
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved SFT student to {output_dir}")


if __name__ == "__main__":
    main()
