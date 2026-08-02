#!/usr/bin/env python
"""Pre-fetch models and datasets used by the labs into the HF cache.

Downloads happen lazily the first time a lab runs anyway, but a 32B/70B
teacher checkpoint mid-lab is a bad surprise, so this script lets you
warm the cache ahead of time for one or more labs.

Examples
--------
    # see what would be downloaded for labs 00-04 without downloading
    python scripts/download_models.py --labs lab00 lab01 lab02 lab03 lab04 --dry-run

    # actually fetch everything needed through the reasoning-distillation labs
    python scripts/download_models.py --labs lab00 lab01 lab02 lab03 lab04 lab05 lab06 lab07

    # fetch the capstone's default (32B) teacher; add --stretch for 70B/72B options
    python scripts/download_models.py --labs lab08 lab09
    python scripts/download_models.py --labs lab09 --stretch
"""
from __future__ import annotations

import argparse
import os

# repo_id -> approximate bf16 download size, just for user-facing display.
SIZE_HINTS_GB = {
    "bert-base-uncased": 0.4,
    "prajjwal1/bert-tiny": 0.02,
    "Qwen/Qwen2.5-0.5B": 1.0,
    "Qwen/Qwen2.5-0.5B-Instruct": 1.0,
    "Qwen/Qwen2.5-1.5B": 3.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 3.0,
    "Qwen/Qwen2.5-3B": 6.0,
    "Qwen/Qwen2.5-7B": 15.0,
    "Qwen/Qwen2.5-7B-Instruct": 15.0,
    "Qwen/Qwen2.5-32B-Instruct": 65.0,
    "Qwen/Qwen2.5-72B-Instruct": 145.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 65.0,
    "meta-llama/Llama-3.1-70B-Instruct": 140.0,
}

MODEL_GROUPS: dict[str, list[str]] = {
    "lab00": ["Qwen/Qwen2.5-0.5B-Instruct"],
    "lab01": ["bert-base-uncased", "prajjwal1/bert-tiny"],
    "lab02": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
    "lab03": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B"],
    "lab04": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B"],
    "lab05": ["Qwen/Qwen2.5-7B"],
    "lab06": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "Qwen/Qwen2.5-1.5B"],
    "lab07": ["Qwen/Qwen2.5-1.5B"],
    "lab08": ["Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"],
    "lab09": ["Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"],
}

# Only pulled in if --stretch is passed alongside lab09.
STRETCH_MODELS = ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"]

# (dataset_name, config_name_or_None)
DATASET_GROUPS: dict[str, list[tuple[str, str | None]]] = {
    "lab01": [("glue", "sst2")],
    "lab02": [("wikitext", "wikitext-2-raw-v1")],
    "lab03": [("tatsu-lab/alpaca", None)],
    "lab04": [("tatsu-lab/alpaca", None)],
    "lab06": [("openai/gsm8k", "main")],
    "lab07": [("openai/gsm8k", "main")],
    "lab08": [("tatsu-lab/alpaca", None), ("openai/gsm8k", "main")],
    "lab09": [
        ("tatsu-lab/alpaca", None),
        ("openai/gsm8k", "main"),
        ("cais/mmlu", "high_school_mathematics"),
        ("cais/mmlu", "elementary_mathematics"),
        ("cais/mmlu", "high_school_physics"),
        ("cais/mmlu", "college_computer_science"),
        ("cais/mmlu", "world_religions"),
    ],
}

ALL_LABS = list(MODEL_GROUPS.keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labs", nargs="+", default=ALL_LABS, choices=ALL_LABS, help="which labs to prefetch for")
    p.add_argument("--stretch", action="store_true", help="also fetch the 70B/72B stretch-goal teachers for lab09")
    p.add_argument("--dry-run", action="store_true", help="print the plan without downloading anything")
    return p.parse_args()


def main():
    args = parse_args()

    models = []
    for lab in args.labs:
        for m in MODEL_GROUPS[lab]:
            if m not in models:
                models.append(m)
    if args.stretch and "lab09" in args.labs:
        for m in STRETCH_MODELS:
            if m not in models:
                models.append(m)

    datasets_needed = []
    for lab in args.labs:
        for d in DATASET_GROUPS.get(lab, []):
            if d not in datasets_needed:
                datasets_needed.append(d)

    total_gb = sum(SIZE_HINTS_GB.get(m, 0.0) for m in models)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    print(f"HF_HOME = {hf_home}")
    print(f"\nModels ({len(models)}, ~{total_gb:.0f}GB total):")
    for m in models:
        print(f"  - {m}  (~{SIZE_HINTS_GB.get(m, '?')}GB)")
    print(f"\nDatasets ({len(datasets_needed)}):")
    for name, config in datasets_needed:
        print(f"  - {name}" + (f" ({config})" if config else ""))

    if args.dry_run:
        print("\n--dry-run set: nothing downloaded.")
        return

    from huggingface_hub import snapshot_download

    print("\nDownloading models...")
    for m in models:
        print(f"  fetching {m} ...")
        snapshot_download(repo_id=m, ignore_patterns=["*.pth", "*.bin.index.json.orig", "*.gguf", "*.pt"])

    print("\nDownloading/caching datasets...")
    from datasets import load_dataset

    for name, config in datasets_needed:
        print(f"  fetching {name} ({config}) ...")
        try:
            load_dataset(name, config) if config else load_dataset(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: failed to prefetch {name}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
