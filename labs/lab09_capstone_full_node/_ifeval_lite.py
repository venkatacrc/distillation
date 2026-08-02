"""A tiny, self-contained set of instruction-following constraints in the
spirit of Google's IFEval benchmark (Zhou et al., "Instruction-Following
Evaluation for Large Language Models," 2023) - not the real benchmark
(which has hundreds of prompts across ~25 verifiable constraint types),
just a handful of programmatically-checkable constraints to get a quick,
reproducible signal on whether distillation preserved instruction-following
precision (as opposed to just "sounding like the teacher").

Kept local to lab09 (not in common/) since it's specific to this capstone's
eval suite, not reused elsewhere in the curriculum.
"""
from __future__ import annotations

from typing import Callable


def _word_count(text: str) -> int:
    return len(text.split())


CONSTRAINTS = [
    {
        "id": "word_count_lt_50",
        "instruction": "Answer in fewer than 50 words: what is the capital of France and why is it historically significant?",
        "check": lambda text: _word_count(text) < 50,
    },
    {
        "id": "no_letter_e",
        "instruction": "Describe what a computer is, without using the letter 'e' anywhere in your answer.",
        "check": lambda text: "e" not in text.lower(),
    },
    {
        "id": "exactly_3_bullet_points",
        "instruction": "List exactly 3 tips for staying healthy, as three separate bullet points, each starting with '-'.",
        "check": lambda text: len([line for line in text.splitlines() if line.strip().startswith("-")]) == 3,
    },
    {
        "id": "all_uppercase",
        "instruction": "Write a one-sentence greeting, entirely in uppercase letters.",
        "check": lambda text: text.strip() == text.strip().upper() and any(c.isalpha() for c in text),
    },
    {
        "id": "ends_with_phrase",
        "instruction": (
            "Explain what photosynthesis is in 2-3 sentences, and end your response with the exact phrase "
            "'Nature is amazing.'"
        ),
        "check": lambda text: "nature is amazing" in text.strip().lower()[-40:],
    },
    {
        "id": "contains_keyword",
        "instruction": "Write a short paragraph about the ocean that must include the word 'turtle' at least once.",
        "check": lambda text: "turtle" in text.lower(),
    },
    {
        "id": "starts_with_word",
        "instruction": "Give one interesting fact about space. Your response must start with the word 'Fact:'.",
        "check": lambda text: text.strip().startswith("Fact:"),
    },
    {
        "id": "no_commas",
        "instruction": "Describe your ideal weekend in 2 sentences without using any commas.",
        "check": lambda text: "," not in text,
    },
]


def score_ifeval_lite(generate_fn: Callable[[list[str]], list[str]]):
    """`generate_fn` takes a list of instruction strings and returns a list
    of response strings. Returns (accuracy, per_constraint_results)."""
    instructions = [c["instruction"] for c in CONSTRAINTS]
    responses = generate_fn(instructions)
    results = []
    for constraint, response in zip(CONSTRAINTS, responses):
        try:
            passed = bool(constraint["check"](response))
        except Exception:
            passed = False
        results.append({"id": constraint["id"], "passed": passed})
    accuracy = sum(r["passed"] for r in results) / len(results)
    return accuracy, results
