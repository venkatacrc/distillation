#!/usr/bin/env python
"""Capstone step 3: on-policy GKD refinement of the SFT-distilled student
(lab04's recipe) on top of step 2's checkpoint - the final stage of the
capstone pipeline, combining offline SFT distillation with on-policy
refinement the way lab08 does at scale."""
from __future__ import annotations

import os
import sys

import torch
import yaml
from datasets import Dataset

LAB_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    import json

    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    gkd_cfg = cfg["gkd"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    train_path = os.path.join(LAB_DIR, "results/teacher_data_train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found - run 01_generate_teacher_data.py first.")
    rows = load_jsonl(train_path)[: gkd_cfg["train_prompts_gkd"]]
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

    sft_path = os.path.join(LAB_DIR, "results/sft_student")
    student_path = sft_path if os.path.exists(sft_path) else cfg["student_base_model"]
    if student_path == cfg["student_base_model"]:
        print(f"{sft_path} not found - initializing GKD directly from base model {cfg['student_base_model']}")

    tokenizer = AutoTokenizer.from_pretrained(student_path)
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16)

    output_dir = os.path.join(LAB_DIR, "results/gkd_student")
    args = GKDConfig(
        output_dir=output_dir,
        per_device_train_batch_size=gkd_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=gkd_cfg["gradient_accumulation_steps"],
        num_train_epochs=gkd_cfg["epochs"],
        learning_rate=gkd_cfg["learning_rate"],
        lmbda=gkd_cfg["lmbda"],
        beta=gkd_cfg["beta"],
        max_new_tokens=gkd_cfg["max_new_tokens"],
        temperature=gkd_cfg["temperature"],
        teacher_model_name_or_path=cfg["teacher_model"],
        teacher_model_init_kwargs={"dtype": torch.bfloat16, "device_map": "auto"},
        seq_kd=False,  # reuse step 1's cached teacher responses as the off-policy target
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = GKDTrainer(model=student, args=args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved final distilled student to {output_dir}")


if __name__ == "__main__":
    main()
