#!/usr/bin/env python
"""Lab01 step 2: train the bert-tiny student with plain hard-label
cross-entropy (no teacher involved). This is the baseline that
train_student_kd.py's distilled students should beat."""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from _shared import evaluate, tokenize_sst2  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_sst2  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)
    torch.manual_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["student_model"], num_labels=2).to(device)

    train_ds_raw = load_sst2("train", n=cfg["train_samples"])
    val_ds_raw = load_sst2("validation", n=cfg["eval_samples"])
    train_ds = tokenize_sst2(tokenizer, train_ds_raw, cfg["max_length"])
    val_ds = tokenize_sst2(tokenizer, val_ds_raw, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["student_batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["student_batch_size"])

    optimizer = AdamW(model.parameters(), lr=cfg["student_lr"])

    print(f"Training baseline student {cfg['student_model']} (hard labels only)...")
    t0 = time.time()
    for epoch in range(cfg["student_epochs"]):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = evaluate(model, val_loader, device)
        print(f"  epoch {epoch + 1}/{cfg['student_epochs']}: val_acc={acc:.4f}")

    final_acc = evaluate(model, val_loader, device)
    print(f"Baseline student final val accuracy: {final_acc:.4f}  ({time.time() - t0:.0f}s)")

    json.dump(
        {"val_accuracy": final_acc, "train_samples": len(train_ds), "eval_samples": len(val_ds)},
        open(os.path.join(LAB_DIR, "results/baseline_metrics.json"), "w"),
        indent=2,
    )
    print("Saved results/baseline_metrics.json")


if __name__ == "__main__":
    main()
