#!/usr/bin/env python
"""Lab05 step 4: compare WikiText-2 perplexity and parameter count across
the original teacher, the pruned-but-not-recovered model, and the
pruned+KD-recovered model."""
from __future__ import annotations

import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_wikitext2  # noqa: E402
from common.eval_harness import perplexity  # noqa: E402
from common.plotting import bar_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_texts = list(load_wikitext2("test", n=cfg["eval"]["perplexity_samples"])["text"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    candidates = [
        ("teacher (original)", cfg["base_model"], False),
        ("pruned (pre-recovery)", os.path.join(LAB_DIR, "results/pruned_model_raw"), True),
        ("pruned + KD recovery", os.path.join(LAB_DIR, "results/recovered_model"), True),
    ]

    results = {}
    for name, path, is_local in candidates:
        if is_local and not os.path.exists(path):
            print(f"Skipping {name}: {path} not found yet (run the earlier steps first)")
            continue
        print(f"Evaluating {name} ({path})...")
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        ppl = perplexity(model, tokenizer, eval_texts, device=device, max_length=cfg["eval"]["max_length"])
        n_params = count_params(model)
        results[name] = {"perplexity": ppl, "n_params": n_params}
        print(f"  perplexity={ppl:.2f}  params={n_params / 1e9:.2f}B")
        del model
        torch.cuda.empty_cache()

    if len(results) < 2:
        print("Not enough models evaluated yet to plot a comparison - run the earlier steps first.")
        return

    bar_plot(
        list(results.keys()),
        [v["perplexity"] for v in results.values()],
        ylabel="perplexity (lower is better)",
        title="WikiText-2 perplexity: teacher vs. pruned vs. pruned+recovered",
        out_path=os.path.join(LAB_DIR, "results/perplexity_comparison.png"),
    )

    json.dump(results, open(os.path.join(LAB_DIR, "results/compression_summary.json"), "w"), indent=2)
    print("\nSaved results/perplexity_comparison.png and results/compression_summary.json")


if __name__ == "__main__":
    main()
