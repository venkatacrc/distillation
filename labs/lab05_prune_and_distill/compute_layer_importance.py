#!/usr/bin/env python
"""Lab05 step 1: rank transformer layers by redundancy using the ShortGPT
"Block Influence" (BI) metric - how much does each layer actually change
its input representation?

    BI_i = E_tokens[ 1 - cosine_similarity(h_in_to_layer_i, h_out_of_layer_i) ]

Layers with low BI barely transform their input - removing them should
perturb the model the least, making them the best pruning candidates.

Reference: Men et al., "ShortGPT: Layers in Large Language Models are More
Redundant Than You Expect," 2024 (https://arxiv.org/abs/2403.03853).
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_wikitext2  # noqa: E402
from common.plotting import bar_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)


@torch.no_grad()
def compute_block_influence(model, tokenizer, texts, max_length, device):
    n_layers = model.config.num_hidden_layers
    bi_sums = torch.zeros(n_layers)
    n_tokens = 0

    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
        if ids.shape[1] < 4:
            continue
        outputs = model(ids, output_hidden_states=True)
        # hidden_states: (n_layers + 1) tensors of [1, seq, dim] - embeddings, then after each layer.
        hidden_states = outputs.hidden_states
        seq_len = ids.shape[1]
        for i in range(n_layers):
            h_in = hidden_states[i][0].float()
            h_out = hidden_states[i + 1][0].float()
            cos_sim = F.cosine_similarity(h_in, h_out, dim=-1)
            bi_sums[i] += (1 - cos_sim).sum().item()
        n_tokens += seq_len

    return (bi_sums / n_tokens).tolist()


def main():
    cfg = yaml.safe_load(open(os.path.join(LAB_DIR, "config.yaml")))
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {cfg['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map=device).eval()

    texts = [t for t in load_wikitext2("train", n=1000)["text"] if len(t.split()) > 30]
    texts = texts[: cfg["importance"]["calibration_samples"]]
    print(f"Computing block influence over {len(texts)} calibration sequences...")
    bi_scores = compute_block_influence(model, tokenizer, texts, cfg["importance"]["max_length"], device)

    ranked = sorted(range(len(bi_scores)), key=lambda i: bi_scores[i])  # ascending: most redundant first
    print("\nLayers ranked from most to least redundant (lowest Block Influence first):")
    for rank, layer_idx in enumerate(ranked):
        print(f"  #{rank:>2}  layer {layer_idx:>2}  BI={bi_scores[layer_idx]:.4f}")

    bar_plot(
        [str(i) for i in range(len(bi_scores))],
        bi_scores,
        ylabel="Block Influence (1 - cosine similarity)",
        title=f"Per-layer redundancy, {cfg['base_model']}",
        out_path=os.path.join(LAB_DIR, "results/block_influence.png"),
        value_labels=False,
    )

    json.dump(
        {"bi_scores": bi_scores, "ranked_most_redundant_first": ranked},
        open(os.path.join(LAB_DIR, "results/layer_importance.json"), "w"),
        indent=2,
    )
    print("\nSaved results/layer_importance.json and results/block_influence.png")


if __name__ == "__main__":
    main()
