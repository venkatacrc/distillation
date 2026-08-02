#!/usr/bin/env python
"""Lab08 step 4: compare baseline / scaled-SFT / scaled-on-policy-GKD
students on held-out prompts, judged by the teacher server over its HTTP
API (same server from serve_teacher_vllm.sh - keep it running for this
step, reusing it as judge as well as data generator)."""
from __future__ import annotations

import gc
import json
import os
import sys

import requests
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.eval_harness import llm_judge_winrate  # noqa: E402
from common.plotting import bar_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def vllm_generate(model_path, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.5, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    outputs = llm.generate(formatted, sampling_params)
    texts = [o.outputs[0].text.strip() for o in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def make_api_judge_fn(base_url: str, model_name: str):
    def judge_generate_fn(prompts):
        outs = []
        for p in prompts:
            payload = {"model": model_name, "messages": [{"role": "user", "content": p}], "max_tokens": 8, "temperature": 0.0}
            r = requests.post(f"{base_url}/chat/completions", json=payload, timeout=60)
            r.raise_for_status()
            outs.append(r.json()["choices"][0]["message"]["content"].strip())
        return outs

    return judge_generate_fn


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    eval_path = os.path.join(LAB_DIR, "results/teacher_responses_eval.jsonl")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"{eval_path} not found - run generate_teacher_responses_api.py first.")
    eval_rows = load_jsonl(eval_path)
    prompts = [r["instruction"] for r in eval_rows]

    print(f"Generating baseline responses from {cfg['student_base_model']}...")
    baseline = vllm_generate(cfg["student_base_model"], prompts)

    candidates = {}
    sft_path = os.path.join(LAB_DIR, "results/sft_student_scaled")
    gkd_path = os.path.join(LAB_DIR, "results/gkd_student_scaled")
    if os.path.exists(sft_path):
        print("Generating scaled-SFT student responses...")
        candidates["sft_scaled"] = vllm_generate(sft_path, prompts)
    if os.path.exists(gkd_path):
        print("Generating scaled on-policy GKD student responses...")
        candidates["gkd_scaled"] = vllm_generate(gkd_path, prompts)

    if not candidates:
        print("Neither results/sft_student_scaled nor results/gkd_student_scaled exist yet - nothing to compare.")
        return

    server = cfg["teacher_server"]
    base_url = f"http://localhost:{server['port']}/v1"
    try:
        requests.get(f"{base_url}/models", timeout=5).raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach the teacher/judge server at {base_url} - is serve_teacher_vllm.sh running? ({exc})") from exc
    judge_fn = make_api_judge_fn(base_url, cfg["judge"]["model"])

    results = {}
    for name, responses in candidates.items():
        result = llm_judge_winrate(judge_fn, prompts, responses, baseline)
        print(f"{name} win-rate vs. baseline student: {result.win_rate_a:.1%} ({result.n_valid}/{result.n_total} decisive)")
        results[name] = result.win_rate_a

    bar_plot(
        list(results.keys()),
        list(results.values()),
        ylabel="win-rate vs. non-distilled baseline student",
        title="Scaled disaggregated pipeline: win-rate vs. baseline",
        out_path=os.path.join(LAB_DIR, "results/scaled_winrates.png"),
    )
    json.dump(results, open(os.path.join(LAB_DIR, "results/scaled_comparison.json"), "w"), indent=2)
    print("Saved results/scaled_winrates.png and results/scaled_comparison.json")


if __name__ == "__main__":
    main()
