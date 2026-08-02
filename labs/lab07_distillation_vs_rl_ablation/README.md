# Lab 07 - Distillation vs. RL Ablation (Reproducing DeepSeek-R1's Table 6)

**Goal:** answer the question DeepSeek-R1's own paper asks: for a *small*
model, is it better to (a) distill reasoning ability from a strong teacher
(lab06), or (b) train it with reinforcement learning directly, the same way
the big teacher itself was trained? This is the one lab in the curriculum
that trains with RL rather than distillation - it exists specifically to
justify, empirically, why the rest of this curriculum bothers with
distillation instead of "just doing RL on everything."

**Hardware:** 1-2 GPUs, ~45-60 minutes. **Prerequisite:** run lab06 first
(its distilled student is the comparison point).

**Papers:** DeepSeek-AI. ["DeepSeek-R1."](https://arxiv.org/abs/2501.12948)
2025, Section 2.4 and Table 6. Shao et al. ["DeepSeekMath: Pushing the
Limits of Mathematical Reasoning in Open Language
Models."](https://arxiv.org/abs/2402.03300) 2024 (introduces GRPO, the RL
algorithm used here and in DeepSeek-R1 itself).

## DeepSeek-R1's own finding

Section 2.4 of the paper states it directly: they tried applying
large-scale RL to `Qwen-32B-Base` directly (calling this "DeepSeek-R1-Zero-Qwen-32B")
and compared it against simply distilling from DeepSeek-R1 into the same
base model. Their Table 6 shows the distilled model substantially
outperforms the RL-trained-from-scratch model on math/reasoning
benchmarks, and their conclusion (Section 2.4, "Distillation vs.
Reinforcement Learning") is blunt:

> "distillation from stronger models ... can achieve better performance
> [than large-scale RL training] with a fraction of the ... compute."

This lab reproduces that comparison at a much smaller scale: same student
base model, one branch trained with GRPO directly on GSM8K, the other
branch being lab06's SFT-distilled model.

## What is GRPO, briefly

Group Relative Policy Optimization (Shao et al. 2024) is the RL algorithm
behind DeepSeek's models. For each prompt, sample a *group* of `G`
completions, score each with a reward function, and compute each
completion's **advantage** as its reward standardized against the group's
own mean/std - no learned value network needed (contrast with PPO). The
policy is updated to increase the probability of above-average completions
in the group and decrease below-average ones, with a KL penalty keeping it
close to a reference policy. Our reward here is as simple as it gets: `1.0`
if the final numeric answer is correct, `0.0` otherwise (implemented with
`common.eval_harness.gsm8k_correct`, the exact same correctness check used
to filter traces in lab06) - a "rule-based reward" in the paper's
terminology, requiring no learned reward model at all.

## Run it

```bash
python train_grpo_baseline.py         # GRPO-trains Qwen2.5-1.5B directly on GSM8K (no teacher at all)
python compare_distillation_vs_rl.py  # evaluates it against lab06's distilled student
```

`config.yaml` deliberately uses the **same base model, seed, and eval
split** as lab06's `config.yaml`, so the two approaches are compared on
identical held-out data.

## What to look for

- `results/distillation_vs_rl_accuracy.png`: at this lab's small scale
  (~200 GRPO steps, ~800 training problems, `Qwen2.5-1.5B`), you should see
  the lab06 distilled student noticeably ahead of the GRPO-from-scratch
  model. RL from a sparse, rule-based reward on a small model with limited
  training is a genuinely hard optimization problem - it has to *discover*
  correct multi-step reasoning through trial and error, whereas
  distillation just has to *imitate* reasoning a much larger, already-competent
  model already demonstrated.
- `results/accuracy_vs_compute.png`: plots accuracy against each method's
  wall-clock training cost (teacher generation + SFT for distillation;
  rollout generation + policy updates for GRPO). Notice that GRPO's cost
  includes *generating full rollouts at every training step* - it's not
  just gradient updates, it's iterative sampling, which is why RL training
  is often far more expensive per unit of learning than offline
  distillation from a dataset generated once.
- This is a small-scale, single-run reproduction - GRPO's sample efficiency
  improves significantly with more training steps, more rollouts per
  prompt (`num_generations`), and better reward shaping, and DeepSeek-R1's
  own actual RL run for `DeepSeek-R1-Zero` used vastly more compute than
  this lab's `max_steps: 200`. The point isn't "GRPO is bad" - it's that
  **for a small model with a limited compute budget, imitating a strong
  teacher's reasoning is a far more sample-efficient way to acquire that
  capability than discovering it from scratch via RL.**

## Next

Lab08 returns to pure distillation, but at a scale where teacher and
student can no longer comfortably share a GPU: a disaggregated pipeline
with the teacher served separately (vLLM tensor-parallel) from student
training (DeepSpeed ZeRO-3), the production pattern for distilling from
genuinely large (32B+) teachers.
