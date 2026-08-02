"""Distillation loss functions shared by lab01, lab02, and lab05.

All divergence functions operate on raw (pre-softmax) logits of shape
``[..., vocab_or_num_classes]`` and reduce over the last dimension, so the
same code works for a classification head (lab01, shape ``[batch, num_classes]``)
or per-token LM logits (lab02/lab05, shape ``[batch, seq_len, vocab_size]``).

References
----------
Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." 2015.
Agarwal et al. "On-Policy Distillation of Language Models: Learning from
    Self-Generated Mistakes." (GKD) 2024.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-9


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "batchmean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unknown reduction: {reduction}")


def soft_ce_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
):
    """Classic Hinton et al. (2015) knowledge distillation loss.

    Blends a hard-label cross-entropy loss against ``labels`` with a
    soft-label distillation loss computed at temperature ``temperature``.
    The soft loss is scaled by ``temperature ** 2`` so that its gradient
    magnitude stays comparable across different choices of T (paper,
    Section 2, footnote 2: "Since the magnitudes of the gradients ...
    scale as 1/T^2 it is important to multiply them by T^2").

    Parameters
    ----------
    student_logits, teacher_logits : [batch, num_classes]
    labels : [batch] integer class indices
    temperature : softmax temperature applied to both distributions
    alpha : weight on the soft (distillation) loss; ``(1 - alpha)`` weight
        goes to the hard label loss.

    Returns
    -------
    (total_loss, hard_loss, soft_loss) — the last two detached, for logging.
    """
    hard_loss = F.cross_entropy(student_logits, labels)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature**2)
    total = alpha * soft_loss + (1 - alpha) * hard_loss
    return total, hard_loss.detach(), soft_loss.detach()


def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dim: int = -1,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(teacher || student).

    Mode-covering: the student is penalized heavily for assigning near-zero
    probability anywhere the teacher assigns non-trivial mass, so the
    student is pushed to "cover" every mode of the teacher's distribution
    (can lead to over-smoothed / blurry students when the teacher is
    highly multimodal). This is what standard response-based KD (Hinton)
    and vanilla sequence-KD approximate.
    """
    log_p_student = F.log_softmax(student_logits, dim=dim)
    p_teacher = F.softmax(teacher_logits, dim=dim)
    kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=dim)
    return _reduce(kl, reduction)


def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dim: int = -1,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(student || teacher).

    Mode-seeking: computed directly from log-softmaxes (rather than via
    ``F.kl_div`` with a probability "target", which would require taking
    log(student_probs) again and can be numerically flaky near 0). The
    student is penalized for putting mass where the teacher has none, so
    it prefers to concentrate on a single high-probability mode of the
    teacher rather than spreading mass thin. This is what on-policy
    distillation methods like GKD (beta=1.0) and MiniLLM optimize, and
    tends to produce more confident, less "hallucinated" students.
    """
    log_p_student = F.log_softmax(student_logits, dim=dim)
    log_p_teacher = F.log_softmax(teacher_logits, dim=dim)
    p_student = log_p_student.exp()
    kl = (p_student * (log_p_student - log_p_teacher)).sum(dim=dim)
    return _reduce(kl, reduction)


def jensen_shannon(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float = 0.5,
    dim: int = -1,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence, as used by TRL's GKDTrainer.

    Interpolates between forward and reverse KL via a mixture distribution
    ``M = beta * teacher + (1 - beta) * student``:

        JSD_beta(student, teacher) = beta * KL(teacher || M) + (1 - beta) * KL(student || M)

    ``beta -> 0`` recovers (a bounded version of) reverse KL behavior,
    ``beta -> 1`` recovers forward KL behavior; ``beta=0.5`` is the
    textbook symmetric JSD. TRL's ``GKDConfig.beta`` uses this exact
    convention.
    """
    p_student = F.softmax(student_logits, dim=dim)
    p_teacher = F.softmax(teacher_logits, dim=dim)
    log_p_student = F.log_softmax(student_logits, dim=dim)
    log_p_teacher = F.log_softmax(teacher_logits, dim=dim)

    p_mix = beta * p_teacher + (1 - beta) * p_student
    log_p_mix = (p_mix + _EPS).log()

    kl_teacher_mix = (p_teacher * (log_p_teacher - log_p_mix)).sum(dim=dim)
    kl_student_mix = (p_student * (log_p_student - log_p_mix)).sum(dim=dim)
    js = beta * kl_teacher_mix + (1 - beta) * kl_student_mix
    return _reduce(js, reduction)


def total_variation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dim: int = -1,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Total variation distance TV(student, teacher) = 0.5 * sum |p_s - p_t|.

    Bounded in [0, 1], symmetric, and satisfies the triangle inequality
    (unlike KL). Useful as a divergence-agnostic sanity metric when
    comparing student/teacher distributions across labs 01/02.
    """
    p_student = F.softmax(student_logits, dim=dim)
    p_teacher = F.softmax(teacher_logits, dim=dim)
    tv = 0.5 * (p_student - p_teacher).abs().sum(dim=dim)
    return _reduce(tv, reduction)


DIVERGENCES = {
    "forward_kl": forward_kl,
    "reverse_kl": reverse_kl,
    "jensen_shannon": jensen_shannon,
    "total_variation": total_variation,
}
