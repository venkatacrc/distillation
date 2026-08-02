#!/usr/bin/env python
"""Lab06 step 2: filter raw traces by correctness (final answer matches
gold) and readability (drop garbled/mixed-language/degenerate traces), then
cap how many traces we keep per problem - mirroring the curation DeepSeek-
R1's distillation dataset went through (Section 2.4: "we ... filter out
chain-of-thought with mixed languages, long paragraphs, and code blocks").
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.eval_harness import gsm8k_correct  # noqa: E402

LAB_DIR = os.path.dirname(__file__)
MAX_KEPT_PER_PROBLEM = 2


def is_readable(trace: str, cfg: dict) -> bool:
    words = trace.split()
    if not (cfg["min_trace_words"] <= len(words) <= cfg["max_trace_words"]):
        return False
    non_ascii = sum(1 for c in trace if ord(c) > 127)
    if len(trace) > 0 and non_ascii / len(trace) > cfg["max_non_ascii_ratio"]:
        return False
    # Crude degeneracy check: a trace that's mostly one repeated line.
    lines = [line.strip() for line in trace.splitlines() if line.strip()]
    if len(lines) > 5 and len(set(lines)) / len(lines) < 0.3:
        return False
    return True


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))["filtering"]
    raw_path = os.path.join(LAB_DIR, "results/raw_traces.jsonl")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found - run generate_traces.py first.")
    rows = load_jsonl(raw_path)

    correct_readable = [r for r in rows if gsm8k_correct(r["trace"], r["gold_answer"]) and is_readable(r["trace"], cfg)]
    print(
        f"{len(correct_readable)}/{len(rows)} raw traces are correct + readable "
        f"({len(correct_readable) / len(rows):.1%})"
    )

    all_problems = {r["question"] for r in rows}
    by_problem = defaultdict(list)
    for r in correct_readable:
        by_problem[r["question"]].append(r)

    kept = []
    for traces in by_problem.values():
        # Prefer shorter correct traces (less rambling), capped per problem
        # so no single easy problem dominates the SFT set.
        traces = sorted(traces, key=lambda r: len(r["trace"].split()))[:MAX_KEPT_PER_PROBLEM]
        kept.extend(traces)

    print(f"{len(by_problem)}/{len(all_problems)} problems have >=1 usable trace")
    print(f"Kept {len(kept)} traces after capping at {MAX_KEPT_PER_PROBLEM} per problem")

    out_path = os.path.join(LAB_DIR, "results/filtered_traces.jsonl")
    with open(out_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    summary = {
        "n_raw": len(rows),
        "n_correct_readable": len(correct_readable),
        "n_problems_total": len(all_problems),
        "n_problems_with_trace": len(by_problem),
        "n_kept": len(kept),
    }
    json.dump(summary, open(os.path.join(LAB_DIR, "results/filter_summary.json"), "w"), indent=2)
    print(f"Saved {len(kept)} filtered traces to {out_path}")


if __name__ == "__main__":
    main()
