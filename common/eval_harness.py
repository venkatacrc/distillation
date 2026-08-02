"""Evaluation utilities shared across labs: GSM8K scoring, MMLU-lite
loglikelihood scoring, pairwise LLM-judge win-rate, and perplexity.

Kept deliberately dependency-light (plain PyTorch + transformers) so it
works whether generations came from a `transformers` pipeline or a vLLM
engine — labs just pass in strings / a callable.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .data import extract_final_number, gsm8k_gold_answer

# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def gsm8k_correct(prediction_text: str, gold_answer_field: str, tol: float = 1e-4) -> bool:
    """Compare a model's free-form prediction against a GSM8K gold answer field."""
    gold = gsm8k_gold_answer(gold_answer_field)
    pred = extract_final_number(prediction_text)
    if gold is None or pred is None:
        return False
    return abs(gold - pred) < tol


def score_gsm8k(predictions: list[str], gold_answer_fields: list[str]) -> dict:
    """Returns {'accuracy': float, 'correct': [bool, ...]} over a batch."""
    correct = [gsm8k_correct(p, g) for p, g in zip(predictions, gold_answer_fields)]
    accuracy = sum(correct) / max(1, len(correct))
    return {"accuracy": accuracy, "correct": correct, "n": len(correct)}


# ---------------------------------------------------------------------------
# MMLU-lite (loglikelihood multiple choice)
# ---------------------------------------------------------------------------

_MMLU_LETTERS = ["A", "B", "C", "D"]


@torch.no_grad()
def mmlu_loglikelihood_choice(
    model,
    tokenizer,
    question: str,
    choices: list[str],
    device: Optional[str] = None,
) -> int:
    """Score each answer choice by the model's total log-likelihood of that
    choice's letter+text continuation, return the argmax index.

    This "loglikelihood scoring" approach (as used by lm-evaluation-harness)
    avoids relying on the model to *generate* a clean "A"/"B"/"C"/"D" token,
    which small un-instruction-tuned students often fail to do.
    """
    device = device or next(model.parameters()).device
    prompt = f"Question: {question}\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(_MMLU_LETTERS, choices)
    ) + "\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    scores = []
    for letter in _MMLU_LETTERS[: len(choices)]:
        continuation = f" {letter}"
        cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
        logits = model(full_ids).logits
        # log p(continuation tokens | prompt)
        cont_len = cont_ids.shape[1]
        log_probs = F.log_softmax(logits[0, -cont_len - 1 : -1], dim=-1)
        token_scores = log_probs.gather(1, cont_ids[0].unsqueeze(1)).squeeze(1)
        scores.append(token_scores.sum().item())
    return int(torch.tensor(scores).argmax().item())


def score_mmlu_lite(model, tokenizer, examples, device: Optional[str] = None) -> dict:
    """``examples`` is an iterable of dicts with 'question', 'choices', 'answer'
    (answer is an int index), e.g. rows from ``common.data.load_mmlu_subset``.
    """
    correct = []
    for ex in examples:
        pred_idx = mmlu_loglikelihood_choice(model, tokenizer, ex["question"], ex["choices"], device)
        correct.append(pred_idx == ex["answer"])
    accuracy = sum(correct) / max(1, len(correct))
    return {"accuracy": accuracy, "correct": correct, "n": len(correct)}


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], device: Optional[str] = None, max_length: int = 1024) -> float:
    device = device or next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n_tokens = ids.shape[1] - 1
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Pairwise LLM-judge win-rate
# ---------------------------------------------------------------------------

_JUDGE_TEMPLATE = """You are an impartial judge evaluating two AI assistant responses to the \
same instruction. Pick the response that is more helpful, correct, and concise.

Instruction:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Answer with a single letter, "A" or "B", and nothing else."""

_LETTER_RE = re.compile(r"\b([AB])\b")


def build_judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return _JUDGE_TEMPLATE.format(prompt=prompt, response_a=response_a, response_b=response_b)


def parse_judge_verdict(judge_output: str) -> Optional[str]:
    match = _LETTER_RE.search(judge_output.strip())
    return match.group(1) if match else None


@dataclass
class JudgeResult:
    win_rate_a: float
    n_valid: int
    n_total: int
    verdicts: list[Optional[str]]


def llm_judge_winrate(
    judge_generate_fn: Callable[[list[str]], list[str]],
    prompts: list[str],
    responses_a: list[str],
    responses_b: list[str],
    debias_position: bool = True,
    seed: int = 0,
) -> JudgeResult:
    """Estimate the win-rate of ``responses_a`` vs ``responses_b`` using a
    judge model.

    ``judge_generate_fn`` takes a list of prompt strings and returns a list
    of generated strings (e.g. a thin wrapper around a vLLM ``LLM.generate``
    call). To reduce position bias, each pair is judged twice with A/B order
    swapped (``debias_position=True``); a verdict only counts as a genuine
    win if both orderings agree.
    """
    judge_prompts = [build_judge_prompt(p, a, b) for p, a, b in zip(prompts, responses_a, responses_b)]
    outputs = judge_generate_fn(judge_prompts)
    verdicts_ab = [parse_judge_verdict(o) for o in outputs]

    if not debias_position:
        wins_a = sum(1 for v in verdicts_ab if v == "A")
        n_valid = sum(1 for v in verdicts_ab if v in ("A", "B"))
        return JudgeResult(
            win_rate_a=wins_a / max(1, n_valid), n_valid=n_valid, n_total=len(prompts), verdicts=verdicts_ab
        )

    swapped_prompts = [build_judge_prompt(p, b, a) for p, a, b in zip(prompts, responses_a, responses_b)]
    swapped_outputs = judge_generate_fn(swapped_prompts)
    verdicts_ba = [parse_judge_verdict(o) for o in swapped_outputs]

    wins_a, n_valid = 0, 0
    combined = []
    for v_ab, v_ba in zip(verdicts_ab, verdicts_ba):
        # In the swapped call, "A" means the original response_b won.
        a_won_first = v_ab == "A"
        a_won_second = v_ba == "B"
        if v_ab in ("A", "B") and v_ba in ("A", "B"):
            n_valid += 1
            if a_won_first and a_won_second:
                wins_a += 1
                combined.append("A")
            elif not a_won_first and not a_won_second:
                combined.append("B")
            else:
                combined.append(None)  # disagreement -> tie, excluded from win count
        else:
            combined.append(None)

    return JudgeResult(win_rate_a=wins_a / max(1, n_valid), n_valid=n_valid, n_total=len(prompts), verdicts=combined)
