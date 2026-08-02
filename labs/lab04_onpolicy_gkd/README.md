# Lab 04 - On-Policy Distillation (GKD)

**Goal:** fix the fundamental limitation of lab03's offline SFT
distillation - the student only ever trains on the teacher's text, never on
its *own* generations - by having the student generate rollouts during
training that the teacher then supervises directly.

**Hardware:** 2 GPUs (teacher + student loaded simultaneously), ~45 minutes.

**Paper:** Agarwal et al. ["On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes."](https://arxiv.org/abs/2306.13649)
(GKD) 2024.

## The problem GKD solves: exposure bias

In lab03, the student is trained via teacher-forcing on teacher-written
text: at every position, it sees the *correct* (teacher's) prefix, even if
its own greedy/sampled prefix would have diverged. At inference time,
though, the student conditions on its *own* previous tokens - so once it
makes one mistake, it's now in a part of input space it never trained on,
and errors compound ("exposure bias" / train-test distribution mismatch).

**On-policy distillation** addresses this directly: let the student
generate a full rollout during training, then have the teacher tell it, at
every position *the student actually visited*, what distribution it should
have produced. The student is now training exactly on the states it will
encounter at inference time.

## How TRL's `GKDTrainer` implements this

`trl.experimental.gkd.GKDTrainer` (wraps `SFTTrainer`) mixes two kinds of
batches per `GKDConfig.lmbda`:

- **Off-policy** (probability `1 - lmbda`): a fixed target sequence (here,
  lab03's cached teacher responses) - basically lab03's setup, but scored
  with a distribution-matching loss instead of a plain cross-entropy loss.
- **On-policy** (probability `lmbda`): the student generates the response
  itself (`max_new_tokens`, `temperature` control this), and the loss
  compares the *teacher's* and *student's* distributions token-by-token
  along the student's own generated sequence.

The per-token loss in both cases is the generalized JS divergence from
lab02 (`common.losses.jensen_shannon`), controlled by `GKDConfig.beta`:
`beta=0.0` behaves like forward KL, `beta=1.0` like reverse KL, `beta=0.5`
is symmetric JSD. If you did lab02 first, this should feel very familiar -
`lmbda` and `beta` are literally the two axes that lab02 taught you to
reason about independently.

## Run it

This lab reuses lab03's cached teacher responses (as the off-policy
target) and, if present, initializes the student from lab03's SFT
checkpoint (an on-policy *refinement* of an already-distilled model, which
is how GKD is used in practice) - **run lab03 first**.

```bash
python train_gkd.py                        # trains 5 (lmbda, beta) configs from config.yaml
python compare_offpolicy_vs_onpolicy.py     # judges each against the lab03 off-policy baseline
```

The default sweep in `config.yaml`:

| lmbda | beta | What it tests |
|---|---|---|
| 0.0 | 0.5 | Fully off-policy (control - should look similar to lab03) |
| 0.5 | 0.5 | Balanced on/off-policy mix, symmetric JSD |
| 1.0 | 0.5 | Fully on-policy (student generates everything) |
| 0.5 | 0.0 | Balanced mix, forward-KL-leaning |
| 0.5 | 1.0 | Balanced mix, reverse-KL-leaning |

## What to look for

- `results/gkd_vs_offpolicy_winrate.png`: the `lmbda=1.0` (fully on-policy)
  and `lmbda=0.5` runs should generally win more often against the pure
  off-policy baseline than the `lmbda=0.0` control (which is essentially
  re-deriving lab03 with a different loss function, so it shouldn't move
  the needle much).
- `results/gkd_response_length.png`: compare the `beta=0.0` (forward-KL-ish)
  vs. `beta=1.0` (reverse-KL-ish) runs. Per lab02's toy experiment,
  forward-KL-style training tends to hedge/cover more possibilities (longer,
  sometimes more repetitive output), while reverse-KL-style training tends
  to commit confidently to a single continuation (often shorter, more
  decisive). This won't be as clean as the toy example - real LLMs have
  billions of "dimensions" of mode structure, not one - but the direction
  of the effect is usually visible.
- Training logs (stdout) show the on-policy runs are slower per step (they
  have to *generate* text before computing a loss on it) - this is the
  real-world cost of on-policy distillation: better sample efficiency in
  terms of avoiding exposure bias, at the cost of more expensive training
  steps (generation is much slower than a single forward pass).

## Next

Lab05 switches from "make a small model behave like a big one" to
"literally make the big model smaller" - structured pruning followed by
distillation-based recovery fine-tuning, the technique behind NVIDIA's
Minitron and similar compact-model recipes.
