#!/usr/bin/env python
"""Lab08 step 1: query the independently-served teacher (see
serve_teacher_vllm.sh) over its OpenAI-compatible HTTP API to generate
responses for the student's training/eval sets.

This is the disaggregated pattern used in production distillation
pipelines: a dedicated, throughput-optimized inference server handles all
teacher generation, fully decoupled from wherever/however student training
runs (here: DeepSpeed ZeRO-3 on a separate set of GPUs, see
train_student_deepspeed.py). Uses plain HTTP (`requests`) rather than the
`openai` client package so it has no dependency beyond what's already
installed for the rest of the stack.

Start serve_teacher_vllm.sh in another terminal first.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys

import requests
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_alpaca  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)

    server = cfg["teacher_server"]
    base_url = f"http://localhost:{server['port']}/v1"
    gen_cfg = cfg["generation"]

    # Fail fast with a clear message if the server isn't up yet.
    try:
        requests.get(f"{base_url}/models", timeout=5).raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the teacher server at {base_url} - is serve_teacher_vllm.sh running? ({exc})"
        ) from exc

    total = cfg["train_prompts"] + cfg["eval_prompts"]
    full_ds = load_alpaca(n=total, seed=cfg["seed"])
    train_ds = full_ds.select(range(cfg["train_prompts"]))
    eval_ds = full_ds.select(range(cfg["train_prompts"], total))

    def query(example):
        user_content = f"{example['instruction']}\n\n{example['input']}" if example.get("input") else example["instruction"]
        payload = {
            "model": cfg["teacher_model"],
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": gen_cfg["max_new_tokens"],
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
        }
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=180)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return {"instruction": user_content, "response": text}

    for split_name, ds in [("train", train_ds), ("eval", eval_ds)]:
        print(f"Querying teacher server for {len(ds)} {split_name} prompts (concurrency={gen_cfg['concurrency']})...")
        with cf.ThreadPoolExecutor(max_workers=gen_cfg["concurrency"]) as pool:
            rows = list(pool.map(query, ds))

        out_path = os.path.join(LAB_DIR, f"results/teacher_responses_{split_name}.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
