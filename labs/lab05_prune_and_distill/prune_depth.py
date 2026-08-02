#!/usr/bin/env python
"""Lab05 step 2: drop the `target_layers_to_prune` most redundant
transformer blocks (per compute_layer_importance.py's ranking) and measure
the resulting perplexity hit *before* any recovery training - this is the
damage distill_recover.py's KD fine-tuning needs to repair.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_wikitext2  # noqa: E402
from common.eval_harness import perplexity  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    importance_path = os.path.join(LAB_DIR, "results/layer_importance.json")
    if not os.path.exists(importance_path):
        raise FileNotFoundError(f"{importance_path} not found - run compute_layer_importance.py first.")
    importance = json.load(open(importance_path))

    n_prune = cfg["target_layers_to_prune"]
    layers_to_drop = sorted(importance["ranked_most_redundant_first"][:n_prune])
    print(f"Dropping {n_prune} layers (by index): {layers_to_drop}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map=device)

    original_n_layers = model.config.num_hidden_layers
    keep_idx = [i for i in range(original_n_layers) if i not in layers_to_drop]
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep_idx])
    model.config.num_hidden_layers = len(keep_idx)

    eval_texts = list(load_wikitext2("test", n=cfg["eval"]["perplexity_samples"])["text"])
    ppl = perplexity(model, tokenizer, eval_texts, device=device, max_length=cfg["eval"]["max_length"])
    print(f"\nPruned model ({len(keep_idx)}/{original_n_layers} layers) perplexity BEFORE recovery: {ppl:.2f}")
    print("(compare against results/compression_summary.json's teacher perplexity after running evaluate_compression.py)")

    out_dir = os.path.join(LAB_DIR, "results/pruned_model_raw")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    json.dump(
        {
            "layers_dropped": layers_to_drop,
            "n_layers_remaining": len(keep_idx),
            "n_layers_original": original_n_layers,
            "perplexity_before_recovery": ppl,
        },
        open(os.path.join(LAB_DIR, "results/prune_summary.json"), "w"),
        indent=2,
    )
    print(f"Saved pruned model to {out_dir} and results/prune_summary.json")


if __name__ == "__main__":
    main()
