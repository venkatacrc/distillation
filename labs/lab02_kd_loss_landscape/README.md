# Lab 02 - The KD Loss Landscape: Forward KL, Reverse KL, and Jensen-Shannon

**Goal:** understand *which* divergence you're minimizing when you do
distribution-matching distillation, and why that choice matters a lot more
for LLMs (huge, highly multimodal next-token distributions) than it did for
the 2-way classifier in lab01.

**Hardware:** 1 GPU, ~25 minutes total.

## The divergences

All four are implemented in [`common/losses.py`](../../common/losses.py)
and operate on raw logits (they apply softmax internally):

- **Forward KL**, \(KL(p_{teacher} \| p_{student})\): the loss Hinton-style
  KD (lab01) and standard sequence-KD approximate. Because
  \(KL(p\|q) = \sum_x p(x) \log \frac{p(x)}{q(x)}\), wherever the teacher
  \(p\) puts non-trivial mass, the student \(q\) is punished heavily (via
  \(\log(1/q(x)) \to \infty\) as \(q(x) \to 0\)) for assigning near-zero
  probability there. Net effect: the student is pushed to **cover every
  mode** of the teacher, even at the cost of spreading probability mass
  into low-density regions between modes ("mode-covering").
- **Reverse KL**, \(KL(p_{student} \| p_{teacher})\): now the student's own
  probability, not the teacher's, weights the sum. The student pays no
  penalty for assigning ~0 probability to regions the teacher cares about
  (that term just vanishes), but pays heavily for putting mass anywhere the
  *teacher* assigns ~0 probability. Net effect: the student **picks one
  mode** of the teacher and commits to it confidently, ignoring the rest
  ("mode-seeking"). This is what on-policy methods like GKD (lab04, with
  `beta=1.0`) and MiniLLM optimize.
- **Jensen-Shannon**, \(JSD_\beta(p, q) = \beta \, KL(p \| M) + (1-\beta)\,
  KL(q \| M)\) with mixture \(M = \beta p + (1-\beta) q\): symmetric,
  bounded in \([0, \log 2]\), and interpolates between the two extremes as
  \(\beta\) varies. This is exactly `GKDConfig.beta`'s convention in TRL.
- **Total variation**, \(TV(p, q) = \tfrac{1}{2}\sum_x |p(x) - q(x)|\):
  bounded, symmetric, satisfies the triangle inequality - a useful
  divergence-agnostic sanity metric.

## Part 1 - Toy distributions: seeing mode-covering vs. mode-seeking

```bash
python toy_distributions.py
```

We define a **bimodal** 1-D teacher (a 50/50 mixture of two Gaussians far
apart) and fit a **student restricted to a single Gaussian** (2 learnable
parameters: mean, std) against it under each divergence. A single Gaussian
*cannot* represent two separate modes, so each divergence is forced to make
a different trade-off, and the difference is visually obvious in
`results/toy_mode_covering_vs_seeking.png`:

- **Forward KL** -> the fitted Gaussian sits *between* the two modes with a
  wide variance, covering both but wasting density on the valley where the
  teacher has ~0 mass (the classic failure mode of maximum-likelihood /
  forward-KL fitting against multimodal data).
- **Reverse KL** -> the fitted Gaussian collapses onto *one* mode with a
  narrow variance, cleanly matching it locally but completely ignoring the
  other mode.
- **Jensen-Shannon** (\(\beta=0.5\)) -> typically lands in between,
  depending on initialization.

This toy example is deliberately not about LLMs - it isolates the pure
mathematical behavior of each divergence when the student's function class
can't represent the teacher exactly, which is *always* true in real
distillation (a 0.5B model is not expressive enough to represent a 7B
model's distribution exactly).

## Part 2 - The same divergences on real LLM next-token distributions

```bash
python token_level_kd_losses.py
```

Loads `Qwen2.5-1.5B` (teacher) and `Qwen2.5-0.5B` (student) - both **base**
models sharing the Qwen2.5 tokenizer/vocab (151,665 tokens), so their
logits live in the same space and can be compared position-by-position. We
run both models over the same WikiText-2 text and compute, at every token
position, the divergence between the teacher's and the (untrained)
student's next-token distribution.

This step does **no training** - it's purely diagnostic, to build intuition
for what these losses look like on real, ~150k-way categorical
distributions before lab04 actually optimizes against them. Look at:

- `results/divergence_comparison.png`: average forward KL, reverse KL, JSD,
  and TV between the two models. Forward KL is usually noticeably larger
  than reverse KL here, because the untrained (relative to the teacher)
  smaller model has a "fatter", less peaked distribution and forward KL
  punishes it hard for spreading mass into regions the sharper teacher
  considers implausible.
- `results/per_token_divergence.png`: per-token divergence across one
  sequence - spikes usually correspond to "surprising"/hard-to-predict
  tokens (rare words, ambiguous continuations) where a much smaller model's
  predictions diverge most from a larger one's.
- The printed top-10 overlap: what fraction of the teacher's top-10 most
  likely next tokens also appear in the student's top-10. This is a cheap,
  interpretable proxy for "how aligned are these two models already,
  before any distillation."

## Why this matters for lab04

Lab04 trains a real student with TRL's `GKDTrainer`, which exposes exactly
the `beta` (JSD interpolation) and `lmbda` (on-policy vs. off-policy data
mix) knobs this lab lets you reason about in isolation first. If you've
internalized "forward KL over-covers, reverse KL under-covers" from this
lab, the `beta` sweep results in lab04 should not be surprising.

## Next

Lab03 puts a loss into an actual training loop for the first time at LLM
scale: response-based / SFT distillation, where the student is trained via
plain next-token cross-entropy on sequences *sampled* from the teacher
(sequence-level KD) rather than matching the teacher's full distribution at
every position.
