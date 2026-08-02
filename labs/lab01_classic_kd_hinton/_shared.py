"""Dataset/training helpers local to lab01 (tokenization + eval loop glue).

Kept separate from the top-level `common` package because it's specific to
this lab's BERT-classification setup, not reused elsewhere.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SST2Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class KDDataset(Dataset):
    """Like SST2Dataset but also carries the teacher's cached logits for
    each example, so shuffling a DataLoader keeps (input, label, teacher
    logits) aligned automatically."""

    def __init__(self, encodings, labels, teacher_logits):
        self.encodings = encodings
        self.labels = labels
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["teacher_logits"] = self.teacher_logits[idx]
        return item


def tokenize_sst2(tokenizer, dataset, max_length: int = 128) -> SST2Dataset:
    enc = tokenizer(
        list(dataset["sentence"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(dataset["label"])
    return SST2Dataset(enc, labels)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


@torch.no_grad()
def collect_logits(model, loader, device) -> torch.Tensor:
    model.eval()
    all_logits = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)
