#!/usr/bin/env python
"""Lab02 part 2: measure how forward KL, reverse KL, Jensen-Shannon, and
total variation behave between two *real* LLMs' next-token distributions -
Qwen2.5-1.5B (larger) vs. Qwen2.5-0.5B (smaller), both base models sharing
one tokenizer/vocab.

No training here - this just characterizes the loss landscape you'd be
optimizing against if you used each divergence for token-level distillation
(that training happens in lab04, via TRL's GKDTrainer).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.data import load_wikitext2  # noqa: E402
from common.losses import forward_kl, jensen_shannon, reverse_kl, total_variation  # noqa: E402
from common.plotting import bar_plot, line_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)
TEACHER = "Qwen/Qwen2.5-1.5B"
STUDENT = "Qwen/Qwen2.5-0.5B"
N_SEQUENCES = 8
MAX_LENGTH = 128
TOP_K = 10


@torch.no_grad()
def get_logits(model, input_ids):
    return model(input_ids).logits[0].float()  # [seq_len, vocab]


def main():
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)  # Qwen2.5 family shares one tokenizer
    print(f"Loading teacher {TEACHER} and student {STUDENT}...")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map=device).eval()
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map=device).eval()

    texts = [t for t in load_wikitext2("train", n=500)["text"] if len(t.split()) > 20][:N_SEQUENCES]

    per_token = {"forward_kl": [], "reverse_kl": [], "jensen_shannon": [], "total_variation": []}
    topk_overlaps = []
    first_seq_len = None

    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        t_logits = get_logits(teacher, ids)
        s_logits = get_logits(student, ids)

        per_token["forward_kl"].extend(forward_kl(s_logits, t_logits, reduction="none").tolist())
        per_token["reverse_kl"].extend(reverse_kl(s_logits, t_logits, reduction="none").tolist())
        per_token["jensen_shannon"].extend(jensen_shannon(s_logits, t_logits, reduction="none").tolist())
        per_token["total_variation"].extend(total_variation(s_logits, t_logits, reduction="none").tolist())

        t_topk = t_logits.topk(TOP_K, dim=-1).indices
        s_topk = s_logits.topk(TOP_K, dim=-1).indices
        topk_overlaps.extend(
            len(set(t_topk[i].tolist()) & set(s_topk[i].tolist())) / TOP_K for i in range(t_topk.shape[0])
        )

        if first_seq_len is None:
            first_seq_len = t_logits.shape[0]

    def summarize(name, values):
        avg = sum(values) / len(values)
        print(f"  {name:<20} mean={avg:.4f}  (n_tokens={len(values)})")
        return avg

    print("\nAverage per-token divergence, student vs. teacher, over WikiText-2:")
    avg = {k: summarize(k, v) for k, v in per_token.items()}
    avg_overlap = summarize(f"top-{TOP_K} overlap", topk_overlaps)

    bar_plot(
        ["forward KL", "reverse KL", "Jensen-Shannon", "total variation"],
        [avg["forward_kl"], avg["reverse_kl"], avg["jensen_shannon"], avg["total_variation"]],
        ylabel="divergence (nats, TV unitless)",
        title=f"{STUDENT} vs {TEACHER}: avg per-token divergence on WikiText-2",
        out_path=os.path.join(LAB_DIR, "results/divergence_comparison.png"),
    )

    n_show = first_seq_len or min(len(per_token["forward_kl"]), MAX_LENGTH)
    line_plot(
        list(range(n_show)),
        {"forward KL": per_token["forward_kl"][:n_show], "reverse KL": per_token["reverse_kl"][:n_show]},
        xlabel="token position (first sequence)",
        ylabel="divergence (nats)",
        title="Per-token divergence across one WikiText-2 sequence",
        out_path=os.path.join(LAB_DIR, "results/per_token_divergence.png"),
    )

    print(f"\nAverage top-{TOP_K} next-token overlap between teacher and (untrained) student: {avg_overlap:.1%}")
    print("Saved results/divergence_comparison.png and results/per_token_divergence.png")


if __name__ == "__main__":
    main()
