# Lab 03 - Response-Based / SFT Distillation

**Goal:** the simplest, most widely used form of LLM distillation in
practice - sample completions from a strong teacher, then fine-tune a
small student on them with plain supervised fine-tuning (SFT). No teacher
logits, no custom loss - just next-token cross-entropy against
teacher-written text.

**Hardware:** 1-2 GPUs, ~45 minutes total.

**Paper:** Kim & Rush. ["Sequence-Level Knowledge
Distillation."](https://arxiv.org/abs/1606.07947) 2016 (this is the
generative-LM analogue of Hinton's lab01 recipe: instead of matching soft
label *distributions*, the student is trained on *samples* drawn from the
teacher).

## Why this works, and its trade-offs

Word/token-level KD (lab02/lab04) requires teacher and student to share a
tokenizer, and requires holding both models' logits (a full vocab-sized
vector per token) - expensive at scale. Response-based/sequence-level
distillation sidesteps both problems entirely:

- Teacher and student can be *any* models, even different tokenizer
  families - you only ever need the teacher's generated *text*.
- Data generation (teacher inference) and student training are fully
  decoupled - generate once with vLLM, train as many student
  configurations as you like against the same cached data.
- The downside: the student only sees one sampled trajectory per prompt
  (not the full distribution over what the teacher *could* have said), so
  it can't learn as much about the teacher's uncertainty. It's also
  strictly **offline** - the student never sees its own mistakes during
  training (that gap is exactly what lab04's on-policy GKD addresses).

## What you'll do

1. **Generate teacher data** (`generate_teacher_responses.py`): batch-generate
   completions for ~2000 Alpaca-style instructions using
   `Qwen2.5-7B-Instruct` via vLLM (fast batched inference), plus a disjoint
   held-out set of 100 prompts for evaluation.
2. **SFT the student** (`train_student_sft.py`): fine-tune `Qwen2.5-0.5B`
   (a *base*, non-instruction-tuned model - it currently can't follow
   instructions at all) on the teacher's (instruction, response) pairs
   using TRL's `SFTTrainer`, with the dataset in the standard
   `messages: [{role, content}, ...]` chat format.
3. **Evaluate** (`evaluate.py`): on the held-out prompts, compare three
   things pairwise with an LLM-judge win-rate: the SFT-distilled student vs.
   the same base model with *no* distillation, and the SFT-distilled
   student vs. the teacher itself.

```bash
python generate_teacher_responses.py
python train_student_sft.py
python evaluate.py
```

## What to look for

- `results/win_rates.png` and `results/eval_results.json`: the
  SFT-distilled student should win the large majority of comparisons
  against the non-distilled baseline (which, being a base model with no
  instruction tuning, often doesn't even attempt to follow the instruction
  format at all) - this is the core "distillation works" result. Its
  win-rate against the teacher itself will be much lower (the student is
  25x smaller and trained on far fewer examples than a real production
  recipe would use), which is expected and fine - the point is the *gap
  closed*, not parity with the teacher.
- Read a few generated responses directly:
  `results/teacher_responses_eval.jsonl` (teacher) vs. generating a couple
  of prompts manually from `results/sft_student` - notice how the student
  picks up the teacher's *style* (formatting, verbosity, hedging phrases)
  even on topics/phrasings it never saw during training.
- Judge self-preference bias is real: because we reuse the teacher as
  judge, "vs. teacher" win-rates are systematically deflated relative to a
  neutral judge. If you have a stronger model handy (e.g. the
  `Qwen2.5-32B-Instruct` used in lab08), swap it into `config.yaml`'s
  `judge.model` and re-run `evaluate.py` to see the difference.

## Next

Lab04 removes the "student never sees its own mistakes" limitation with
**on-policy distillation (GKD)**: the student generates its own rollouts
during training, and the teacher supervises them directly, closing the
train/inference distribution mismatch that pure offline SFT distillation
leaves on the table.
