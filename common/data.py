"""Dataset loading and formatting helpers shared across labs.

Keeps every lab's data plumbing (chat templates, answer extraction, subset
sizing) in one place so labs stay focused on the distillation technique
being taught. All loaders accept an ``n`` argument to cap dataset size for
fast iteration.
"""
from __future__ import annotations

import re
from typing import Optional

from datasets import Dataset, load_dataset

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_sst2(split: str = "train", n: Optional[int] = None) -> Dataset:
    """Stanford Sentiment Treebank v2 (binary sentiment), used in lab01."""
    ds = load_dataset("glue", "sst2", split=split)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def load_alpaca(n: Optional[int] = None, seed: int = 0) -> Dataset:
    """Stanford Alpaca instruction-following dataset, used in labs 03/04/08/09."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def load_gsm8k(split: str = "train", n: Optional[int] = None, seed: int = 0) -> Dataset:
    """Grade-school math word problems, used in labs 06/07/09."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def load_wikitext2(split: str = "train", n: Optional[int] = None) -> Dataset:
    """WikiText-2 raw text, used in lab02 for real token-distribution comparisons."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.filter(lambda x: len(x["text"].strip()) > 20)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return ds


DEFAULT_MMLU_SUBJECTS = [
    "high_school_mathematics",
    "elementary_mathematics",
    "high_school_physics",
    "college_computer_science",
    "world_religions",
]


def load_mmlu_subset(
    subjects: Optional[list[str]] = None,
    n_per_subject: int = 40,
    split: str = "test",
) -> Dataset:
    """A small, fast-to-run slice of MMLU across a handful of subjects.

    Used as the "MMLU-lite" general-knowledge eval in labs 08/09 - not a
    substitute for the full 57-subject benchmark, just a quick signal.
    """
    subjects = subjects or DEFAULT_MMLU_SUBJECTS
    rows = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split=split)
        ds = ds.select(range(min(n_per_subject, len(ds))))
        for ex in ds:
            rows.append({**ex, "subject": subject})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def alpaca_to_messages(example: dict) -> list[dict]:
    """Convert one Alpaca-style example into a chat ``messages`` list.

    Matches the ``messages: [{role, content}, ...]`` format expected by
    ``trl.SFTTrainer`` and ``trl.experimental.gkd.GKDTrainer``.
    """
    if example.get("input"):
        user_content = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_content = example["instruction"]
    messages = [{"role": "user", "content": user_content}]
    if example.get("output"):
        messages.append({"role": "assistant", "content": example["output"]})
    return messages


def gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem. Think step by step, "
        "then give the final answer on its own line in the form "
        "'#### <number>'.\n\n"
        f"Question: {question}\nAnswer:"
    )


_GSM8K_GOLD_RE = re.compile(r"####\s*([\-\$0-9,\.]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _clean_number(text: str) -> Optional[float]:
    text = text.strip().replace(",", "").replace("$", "").rstrip(".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def gsm8k_gold_answer(answer_field: str) -> Optional[float]:
    """Extract the gold numeric answer from a GSM8K ``answer`` field (ends in '#### N')."""
    match = _GSM8K_GOLD_RE.search(answer_field)
    if not match:
        return None
    return _clean_number(match.group(1))


def extract_final_number(text: str) -> Optional[float]:
    """Best-effort extraction of a model's final numeric answer.

    Checks, in priority order: an explicit ``#### N`` marker (the format we
    ask students to produce), a ``\\boxed{N}`` marker (common in
    DeepSeek-R1-style CoT output), then falls back to the last number that
    appears anywhere in the text.
    """
    match = _GSM8K_GOLD_RE.search(text)
    if match:
        val = _clean_number(match.group(1))
        if val is not None:
            return val
    match = _BOXED_RE.findall(text)
    if match:
        val = _clean_number(match[-1])
        if val is not None:
            return val
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _clean_number(numbers[-1])
    return None
