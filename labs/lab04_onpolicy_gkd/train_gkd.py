#!/usr/bin/env python
"""Lab04: on-policy distillation with TRL's GKDTrainer.

Trains the student to match the teacher's next-token distribution using a
mix of (a) off-policy target sequences (reusing lab03's cached teacher
responses) and (b) on-policy sequences the student generates itself
*during* training, with the teacher scoring both. Sweeps `lmbda` (the
on-policy/off-policy data mix) and `beta` (forward vs. reverse KL
interpolation, see lab02) as defined in config.yaml.

Reference: Agarwal et al., "On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes" (GKD), 2024.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LAB_DIR = os.path.dirname(__file__)
LAB03_RESULTS = os.path.join(LAB_DIR, "..", "lab03_response_based_sft_distillation", "results")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_dataset(split: str, n: int | None = None) -> Dataset:
    path = os.path.join(LAB03_RESULTS, f"teacher_responses_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run lab03's generate_teacher_responses.py first "
            "(lab04 reuses its cached teacher responses as the off-policy target)."
        )
    rows = load_jsonl(path)
    if n:
        rows = rows[:n]
    return Dataset.from_list(
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


def student_init_path(cfg) -> str:
    sft_path = os.path.join(LAB03_RESULTS, "sft_student")
    if os.path.exists(sft_path):
        print(f"Initializing student from lab03's SFT checkpoint: {sft_path}")
        return sft_path
    print(f"lab03 SFT checkpoint not found - initializing student from base model {cfg['student_base_model']}")
    return cfg["student_base_model"]


def run_one(cfg, train_dataset, eval_dataset, lmbda: float, beta: float, run_name: str) -> str:
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    student_path = student_init_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(student_path)
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16)

    gkd_cfg = cfg["gkd"]
    output_dir = os.path.join(LAB_DIR, "results", run_name)
    args = GKDConfig(
        output_dir=output_dir,
        per_device_train_batch_size=gkd_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=gkd_cfg["gradient_accumulation_steps"],
        num_train_epochs=gkd_cfg["epochs"],
        learning_rate=gkd_cfg["learning_rate"],
        lmbda=lmbda,
        beta=beta,
        max_new_tokens=gkd_cfg["max_new_tokens"],
        temperature=gkd_cfg["temperature"],
        teacher_model_name_or_path=cfg["teacher_model"],
        teacher_model_init_kwargs={"dtype": torch.bfloat16},
        seq_kd=False,  # use the dataset's cached teacher responses as the off-policy target
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = GKDTrainer(
        model=student,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved GKD student (lmbda={lmbda}, beta={beta}) to {output_dir}")
    return output_dir


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    train_dataset = build_dataset("train", n=cfg["train_prompts"])
    eval_dataset = build_dataset("eval", n=cfg["eval_prompts"]) if cfg.get("eval_prompts") else None

    manifest = []
    for run in cfg["sweep"]:
        run_name = f"gkd_lmbda{run['lmbda']}_beta{run['beta']}"
        print(f"\n=== Training {run_name} ===")
        out_dir = run_one(cfg, train_dataset, eval_dataset, run["lmbda"], run["beta"], run_name)
        manifest.append({"lmbda": run["lmbda"], "beta": run["beta"], "output_dir": out_dir})

    json.dump(manifest, open(os.path.join(LAB_DIR, "results/sweep_manifest.json"), "w"), indent=2)
    print("\nSaved results/sweep_manifest.json")


if __name__ == "__main__":
    main()
