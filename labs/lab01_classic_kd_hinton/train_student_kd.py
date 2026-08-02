#!/usr/bin/env python
"""Lab01 step 3: train the bert-tiny student with Hinton-style knowledge
distillation (common.losses.soft_ce_kd_loss), sweeping the softmax
temperature T and the hard/soft blend weight alpha.

Requires results/teacher_train_logits.pt from train_teacher.py.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from _shared import KDDataset, evaluate, tokenize_sst2  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_sst2  # noqa: E402
from common.losses import soft_ce_kd_loss  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def train_one(cfg, tokenizer, train_ds_raw, teacher_logits, val_loader, device, temperature, alpha):
    torch.manual_seed(cfg["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["student_model"], num_labels=2).to(device)

    enc = tokenizer(
        list(train_ds_raw["sentence"]),
        padding="max_length",
        truncation=True,
        max_length=cfg["max_length"],
        return_tensors="pt",
    )
    labels = torch.tensor(train_ds_raw["label"])
    ds = KDDataset(enc, labels, teacher_logits)
    loader = DataLoader(ds, batch_size=cfg["student_batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=cfg["student_lr"])
    for _ in range(cfg["student_epochs"]):
        model.train()
        for batch in loader:
            t_logits = batch.pop("teacher_logits").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_ = batch.pop("labels")
            s_logits = model(**batch).logits
            loss, _, _ = soft_ce_kd_loss(s_logits, t_logits, labels_, temperature=temperature, alpha=alpha)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return evaluate(model, val_loader, device)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"])
    train_ds_raw = load_sst2("train", n=cfg["train_samples"])
    val_ds_raw = load_sst2("validation", n=cfg["eval_samples"])
    val_ds = tokenize_sst2(tokenizer, val_ds_raw, cfg["max_length"])
    val_loader = DataLoader(val_ds, batch_size=cfg["student_batch_size"])

    logits_path = os.path.join(LAB_DIR, "results/teacher_train_logits.pt")
    if not os.path.exists(logits_path):
        raise FileNotFoundError(f"{logits_path} not found - run train_teacher.py first.")
    teacher_logits = torch.load(logits_path)
    assert len(teacher_logits) == len(train_ds_raw), (
        "Cached teacher logits don't match train_samples in config.yaml - "
        "re-run train_teacher.py after changing train_samples."
    )

    sweep_results = []
    for temperature in cfg["temperatures"]:
        for alpha in cfg["alphas"]:
            t0 = time.time()
            acc = train_one(cfg, tokenizer, train_ds_raw, teacher_logits, val_loader, device, temperature, alpha)
            dt = time.time() - t0
            print(f"T={temperature:>4} alpha={alpha:>4}  val_acc={acc:.4f}  ({dt:.0f}s)")
            sweep_results.append({"temperature": temperature, "alpha": alpha, "val_accuracy": acc})
            if alpha == 0.0:
                # Pure hard-label training doesn't depend on temperature at
                # all - don't waste time re-running identical configs.
                break

    json.dump(sweep_results, open(os.path.join(LAB_DIR, "results/kd_sweep.json"), "w"), indent=2)
    print("Saved results/kd_sweep.json")


if __name__ == "__main__":
    main()
