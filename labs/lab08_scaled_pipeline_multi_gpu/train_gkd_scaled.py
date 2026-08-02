#!/usr/bin/env python
"""Lab08 step 3: on-policy GKD refinement of the scaled SFT student,
re-running lab04's recipe at this lab's larger teacher/student scale.

Note on disaggregation: TRL's GKDTrainer needs the teacher's weights
in-process (it calls the teacher directly inside the training step to
score the student's on-policy rollouts), so this step - unlike step 1/2's
HTTP-served data generation - loads the teacher directly rather than
querying serve_teacher_vllm.sh. `device_map="auto"` spreads the 32B
teacher across whichever GPUs are visible; run this after stopping the
vLLM server (or on a separate set of free GPUs) so the teacher has room.
Fully disaggregating an on-policy loop like this over HTTP is possible in
principle (score the student's exact sampled tokens via a teacher-side
scoring/logprobs endpoint) but is meaningfully more complex and fragile
than the sequence-level generation used in step 1 - a good next project if
you want to go further than this curriculum.
"""
from __future__ import annotations

import os
import sys

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import alpaca_to_messages, load_alpaca  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    gkd_cfg = cfg["gkd"]

    from trl.experimental.gkd import GKDConfig, GKDTrainer

    student_path = os.path.join(LAB_DIR, "results/sft_student_scaled")
    if not os.path.exists(student_path):
        print(f"{student_path} not found - initializing from base model {cfg['student_base_model']} instead")
        student_path = cfg["student_base_model"]

    tokenizer = AutoTokenizer.from_pretrained(student_path)
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16)

    train_ds_raw = load_alpaca(n=gkd_cfg["train_prompts_gkd"], seed=cfg["seed"])
    train_dataset = train_ds_raw.map(
        lambda ex: {"messages": alpaca_to_messages(ex)}, remove_columns=train_ds_raw.column_names
    )

    output_dir = os.path.join(LAB_DIR, "results/gkd_student_scaled")
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
        seq_kd=True,  # let the teacher generate off-policy targets on the fly instead of using cached data
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = GKDTrainer(model=student, args=args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved scaled on-policy GKD student to {output_dir}")


if __name__ == "__main__":
    main()
