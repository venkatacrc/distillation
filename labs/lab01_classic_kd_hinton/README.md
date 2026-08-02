# Lab 01 - Classic Knowledge Distillation (Hinton et al. 2015)

**Goal:** implement the original KD loss from scratch and see, empirically,
that a tiny student trained with *soft* teacher labels beats the same
student trained on *hard* ground-truth labels alone.

**Hardware:** 1 GPU, ~20 minutes total.

**Paper:** Hinton, Vinyals, Dean. ["Distilling the Knowledge in a Neural
Network."](https://arxiv.org/abs/1503.02531) 2015.

## The idea

A trained classifier's raw output logits, once you look past the single
predicted class, encode rich information: how *confident* the model is, and
which wrong classes it considers "almost right" (Hinton calls this "dark
knowledge"). A one-hot hard label throws all of that away.

KD trains a student to match the teacher's full output distribution instead
of just the argmax label. Concretely, given teacher logits \(z_t\) and
student logits \(z_s\) for the same input:

1. Soften both distributions with a temperature \(T > 1\):
   \[ p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \]
   Higher \(T\) flattens the distribution, revealing more of the relative
   probabilities the teacher assigns to non-target classes.
2. Minimize \(KL(p_t \,\|\, p_s)\) between the softened teacher and student
   distributions (the "soft loss").
3. Blend this with the usual hard-label cross-entropy loss:
   \[ \mathcal{L} = \alpha \cdot T^2 \cdot KL(p_t \| p_s) + (1-\alpha) \cdot \text{CE}(y, z_s) \]
   The \(T^2\) factor matters: gradients of the soft loss w.r.t. logits
   scale as \(1/T^2\), so without it, increasing \(T\) would silently shrink
   the soft loss's effective weight (paper, Section 2).

This is implemented exactly as `common.losses.soft_ce_kd_loss(student_logits,
teacher_logits, labels, temperature, alpha)` - read that function, it's ~10
lines.

## Setup for this lab

- **Task:** SST-2 binary sentiment classification (from GLUE).
- **Teacher:** `bert-base-uncased` (110M params), fine-tuned by you in step 1.
- **Student:** `prajjwal1/bert-tiny` (4.4M params - 2 layers, 128 hidden dim),
  about **25x smaller** than the teacher.
- Because this is classification (a fixed 2-way softmax), teacher and
  student don't need to share a tokenizer/vocabulary the way generative
  LLM distillation does (that constraint shows up starting in lab02).

## Run it

```bash
python train_teacher.py          # fine-tunes BERT-base, caches its logits over the train set
python train_student_baseline.py # trains bert-tiny with hard labels only (the baseline to beat)
python train_student_kd.py       # trains bert-tiny with KD, sweeping temperature T and blend alpha
python plot_results.py           # writes results/*.png
```

All outputs land in `results/`: `teacher_metrics.json`,
`baseline_metrics.json`, `kd_sweep.json`, `teacher_train_logits.pt`
(cached soft labels), and two PNGs.

## What to look for

- `results/teacher_vs_student.png` should show a clear ordering: teacher >
  best-KD student > baseline student. The gap between "baseline" and "best
  KD" is entirely attributable to the extra information in the teacher's
  soft labels - same architecture, same data, same number of epochs.
- `results/accuracy_vs_temperature.png` shows accuracy vs. \(T\) for each
  \(\alpha\). Typically there's a sweet spot around \(T \in [2, 4]\):
  \(T=1\) barely softens the teacher's already-confident predictions (little
  extra signal over hard labels), while very large \(T\) over-flattens the
  distribution into near-uniform noise.
- \(\alpha=0\) is mathematically identical to the baseline script (pure hard
  label loss) - the sweep script skips redundant temperature values for
  \(\alpha=0\) and the plot script cross-checks the two numbers agree.
- Try \(\alpha=1\) (pure soft loss, no ground truth at all) - it usually
  still works reasonably well, since the teacher's soft labels already
  correlate strongly with the true label; this is a preview of "sequence-level"
  distillation in lab03, where the student never sees ground-truth
  labels either, only teacher output.

## Next

Lab02 dissects *which* divergence you use to compare distributions (forward
KL vs. reverse KL vs. Jensen-Shannon) and why that choice matters much more
once the "distribution" being matched is a full vocabulary-sized next-token
distribution rather than a 2-way class softmax.
