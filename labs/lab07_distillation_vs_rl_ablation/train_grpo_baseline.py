#!/usr/bin/env python
"""Lab07 step 1: train the *same* small base model used in lab06 directly
with GRPO reinforcement learning on GSM8K, using a simple rule-based
reward (1.0 if the final answer is correct, else 0.0) - no teacher, no
distillation involved at all.

This reproduces the "large-scale RL directly on the small model" side of
DeepSeek-R1's own ablation (paper Table 6): they found this is *much* less
effective, and far more compute-hungry, than distilling from a strong
teacher (lab06). compare_distillation_vs_rl.py checks whether that holds
at this lab's much smaller scale.

References: DeepSeek-AI, "DeepSeek-R1", 2025, Section 2.4 / Table 6.
Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
Open Language Models" (introduces GRPO), 2024.
"""
from __future__ import annotations

import json
import os
import sys
import time

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import gsm8k_prompt, load_gsm8k  # noqa: E402
from common.eval_harness import gsm8k_correct  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def gsm8k_reward(completions, gold_answer, **kwargs):
    """GRPO reward functions receive the batch's completions plus any extra
    dataset columns as kwargs (here, `gold_answer`), and must return a list
    of floats, one per completion."""
    return [1.0 if gsm8k_correct(c, g) else 0.0 for c, g in zip(completions, gold_answer)]


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    train_ds_raw = load_gsm8k("train", n=cfg["train_problems"], seed=cfg["seed"])
    train_dataset = Dataset.from_list(
        [{"prompt": gsm8k_prompt(ex["question"]), "gold_answer": ex["answer"]} for ex in train_ds_raw]
    )

    grpo_cfg = cfg["grpo"]
    output_dir = os.path.join(LAB_DIR, "results/grpo_student")
    args = GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo_cfg["num_generations"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        max_steps=grpo_cfg["max_steps"],
        learning_rate=grpo_cfg["learning_rate"],
        max_prompt_length=grpo_cfg["max_prompt_length"],
        max_completion_length=grpo_cfg["max_completion_length"],
        temperature=grpo_cfg["temperature"],
        use_vllm=True,
        vllm_mode="colocate",
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=cfg["student_base_model"],
        reward_funcs=gsm8k_reward,
        args=args,
        train_dataset=train_dataset,
    )

    print(
        f"Training {cfg['student_base_model']} with GRPO for {grpo_cfg['max_steps']} steps "
        f"({grpo_cfg['num_generations']} rollouts/prompt, rule-based GSM8K reward)..."
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(output_dir)
    print(f"GRPO training done in {elapsed:.0f}s ({elapsed / 3600:.2f} GPU-hours on this run's GPU count)")

    json.dump(
        {
            "elapsed_seconds": elapsed,
            "max_steps": grpo_cfg["max_steps"],
            "num_generations": grpo_cfg["num_generations"],
        },
        open(os.path.join(LAB_DIR, "results/grpo_train_summary.json"), "w"),
        indent=2,
    )
    print(f"Saved GRPO-trained model to {output_dir}")


if __name__ == "__main__":
    main()
