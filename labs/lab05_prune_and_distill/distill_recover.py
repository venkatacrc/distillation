#!/usr/bin/env python
"""Lab05 step 3: recover the pruned model's quality via knowledge
distillation against the original, uncompressed model as teacher - this is
the "distillation" half of the Minitron-style prune-then-distill recipe.

Launch across multiple GPUs (teacher is replicated per-process; the
student is trained with standard data-parallel gradient sync):

    accelerate launch --multi_gpu --num_processes 4 distill_recover.py

(or just `python distill_recover.py` for a single-GPU run)
"""
from __future__ import annotations

import os
import sys
import time

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_wikitext2  # noqa: E402
from common.losses import forward_kl  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    rec_cfg = cfg["recovery"]
    accelerator = Accelerator()
    device = accelerator.device

    pruned_path = os.path.join(LAB_DIR, "results/pruned_model_raw")
    if not os.path.exists(pruned_path):
        raise FileNotFoundError(f"{pruned_path} not found - run prune_depth.py first.")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print(f"Loading pruned student from {pruned_path} and teacher {cfg['base_model']}...")
    student = AutoModelForCausalLM.from_pretrained(pruned_path, dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    texts = list(load_wikitext2("train", n=rec_cfg["train_samples"])["text"])
    enc = tokenizer(texts, truncation=True, max_length=rec_cfg["max_length"], padding="max_length", return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(dataset, batch_size=rec_cfg["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=rec_cfg["learning_rate"])
    n_steps = len(loader) * rec_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * n_steps), num_training_steps=n_steps)

    student, optimizer, loader, scheduler = accelerator.prepare(student, optimizer, loader, scheduler)

    alpha = rec_cfg["kd_alpha"]
    step = 0
    t0 = time.time()
    for epoch in range(rec_cfg["epochs"]):
        student.train()
        for input_ids, attention_mask in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            outputs = student(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            hard_loss = outputs.loss
            # Compare next-token distributions position-by-position (forward KL,
            # mode-covering - see lab02) between teacher and pruned student.
            kd_loss = forward_kl(
                outputs.logits[:, :-1].float(), teacher_logits[:, :-1].float().to(outputs.logits.device)
            )
            loss = alpha * kd_loss + (1 - alpha) * hard_loss

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % rec_cfg["log_every"] == 0 and accelerator.is_main_process:
                print(f"epoch {epoch} step {step}/{n_steps}: hard_loss={hard_loss.item():.3f} kd_loss={kd_loss.item():.3f}")
            step += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\nRecovery training done in {time.time() - t0:.0f}s")
        out_dir = os.path.join(LAB_DIR, "results/recovered_model")
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"Saved recovered model to {out_dir}")


if __name__ == "__main__":
    main()
