#!/usr/bin/env python
"""Lab01 step 1: fine-tune BERT-base (the teacher) on SST-2 sentiment
classification, then cache its logits over the training set.

We cache the logits once because this lab does *offline* distillation: the
teacher is frozen after this step, so there's no need to re-run its forward
pass every time we train a student with a different (temperature, alpha).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from _shared import collect_logits, evaluate, tokenize_sst2  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_sst2  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)
    torch.manual_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["teacher_model"], num_labels=2).to(device)

    train_ds_raw = load_sst2("train", n=cfg["train_samples"])
    val_ds_raw = load_sst2("validation", n=cfg["eval_samples"])

    train_ds = tokenize_sst2(tokenizer, train_ds_raw, cfg["max_length"])
    val_ds = tokenize_sst2(tokenizer, val_ds_raw, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["teacher_batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["teacher_batch_size"])
    # Unshuffled pass over the training set so cached logits line up 1:1
    # with `train_ds_raw`'s row order (train_student_kd.py relies on this).
    train_loader_ordered = DataLoader(train_ds, batch_size=cfg["teacher_batch_size"], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=cfg["teacher_lr"])
    n_steps = len(train_loader) * cfg["teacher_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps
    )

    print(f"Fine-tuning teacher {cfg['teacher_model']} on {len(train_ds)} SST-2 examples...")
    t0 = time.time()
    for epoch in range(cfg["teacher_epochs"]):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        acc = evaluate(model, val_loader, device)
        print(f"  epoch {epoch + 1}/{cfg['teacher_epochs']}: val_acc={acc:.4f}")

    final_acc = evaluate(model, val_loader, device)
    print(f"Teacher final val accuracy: {final_acc:.4f}  ({time.time() - t0:.0f}s)")

    print("Caching teacher logits over the training set for offline KD...")
    teacher_logits = collect_logits(model, train_loader_ordered, device)
    torch.save(teacher_logits, os.path.join(LAB_DIR, "results/teacher_train_logits.pt"))

    model.save_pretrained(os.path.join(LAB_DIR, "results/teacher_model"))
    tokenizer.save_pretrained(os.path.join(LAB_DIR, "results/teacher_model"))

    json.dump(
        {"val_accuracy": final_acc, "train_samples": len(train_ds), "eval_samples": len(val_ds)},
        open(os.path.join(LAB_DIR, "results/teacher_metrics.json"), "w"),
        indent=2,
    )
    print("Saved results/teacher_metrics.json, results/teacher_train_logits.pt, results/teacher_model/")


if __name__ == "__main__":
    main()
