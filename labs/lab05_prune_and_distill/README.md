# Lab 05 - Structured Pruning + Distillation Recovery

**Goal:** build a small model *from* a big one by physically removing the
most redundant transformer layers, then use knowledge distillation against
the original uncompressed model to recover most of the quality lost in
pruning. This is the recipe behind NVIDIA's Minitron family and similar
"prune-then-distill" compact models.

**Hardware:** 2-4 GPUs, ~60 minutes.

**Papers:** Men et al. ["ShortGPT: Layers in Large Language Models are More
Redundant Than You Expect."](https://arxiv.org/abs/2403.03853) 2024 (the
layer-importance metric used here). Muralidharan et al. ["Compact Language
Models via Pruning and Knowledge
Distillation."](https://arxiv.org/abs/2407.14679) (Minitron) 2024 (the
overall prune-then-distill recipe, at much larger scale than this lab).

## The recipe, in three steps

Unlike labs 01-04 (train a *separately initialized* small model to imitate
a big one), this lab starts from the big model's own weights:

1. **Rank layers by redundancy** (`compute_layer_importance.py`): for each
   transformer block, measure how much it actually changes its input
   representation, using ShortGPT's **Block Influence** metric:
   \[ BI_i = \mathbb{E}_{\text{tokens}}\left[\,1 - \cos\big(h_{\text{in},i},\, h_{\text{out},i}\big)\,\right] \]
   A layer with `BI ≈ 0` leaves its input almost unchanged (its residual
   branch could be replaced with a no-op with little consequence); a layer
   with high `BI` is doing meaningful transformation work. Modern
   transformers are surprisingly redundant - ShortGPT found many models
   have several layers you can delete with only a small quality hit.
2. **Drop the most redundant layers** (`prune_depth.py`): physically remove
   the lowest-`BI` transformer blocks from `Qwen2.5-7B` (default: 6 of 28
   layers, ~21%) and measure the immediate perplexity damage - this is the
   quality gap the next step needs to close.
3. **Recover via distillation** (`distill_recover.py`): fine-tune the
   pruned model against the *original, uncompressed* model as a frozen
   teacher, using a blend of the normal next-token LM loss and a
   **forward-KL** distribution-matching loss (`common.losses.forward_kl`,
   from lab02) between teacher and (pruned) student logits at every
   position. This is feature/logit-based distillation used purely for
   *recovery*, not for teaching a new capability.

```bash
python compute_layer_importance.py
python prune_depth.py
accelerate launch --multi_gpu --num_processes 4 distill_recover.py   # or: python distill_recover.py for 1 GPU
python evaluate_compression.py
```

## Multi-GPU setup

`distill_recover.py` uses HF `accelerate` for data-parallel training: the
frozen teacher (7B, ~15GB in bf16) and the pruned student (~11GB with 6/28
layers removed) are both replicated on every GPU, and gradients for the
student are synced across processes the standard DDP way. This isn't about
sharding a model that's too big for one GPU (a 7B model fits easily on a
single B200) - it's about using multiple GPUs' throughput to get through
the recovery fine-tuning corpus faster, which matters more as you scale
this recipe to bigger base models (lab08/09 use a similar multi-GPU
pattern for a very different reason: the *teacher* itself is too big to
casually run on the same GPU as the student).

## What to look for

- `results/block_influence.png`: you'll typically see the **first few and
  last few layers** have high Block Influence (they do the most
  representation-building/task-specific work), while several **middle
  layers** have surprisingly low BI - these are ShortGPT's pruning
  candidates. This pattern replicates across many transformer families.
- `results/prune_summary.json` vs. `results/compression_summary.json`:
  compare "perplexity before recovery" (right after deleting layers, no
  training at all) against "pruned + KD recovery" - recovery training
  should claw back a substantial fraction of the quality gap, even with
  only ~4000 training sequences and 1 epoch, since we're only asking the
  remaining layers to compensate for a well-chosen set of missing ones, not
  learn a wholly new function.
- `results/perplexity_comparison.png`: the final 3-way comparison. The
  recovered model won't fully match the teacher (you removed 21% of its
  layers!) but should be much closer to it than the raw pruned model, at a
  fraction of the parameter count and inference cost.
- Try changing `target_layers_to_prune` in `config.yaml` - there's a
  sharp knee in most models' pruning curves where quality degrades much
  faster once you remove too many layers, even with recovery training.

## Next

Lab06 shifts from *compression* to *capability transfer*: distilling a
specific skill (multi-step mathematical reasoning) from a much larger
reasoning-specialized teacher into a small model, reproducing DeepSeek-R1's
own distillation recipe.
