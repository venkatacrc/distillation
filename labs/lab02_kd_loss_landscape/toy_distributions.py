#!/usr/bin/env python
"""Lab02 part 1: forward KL vs. reverse KL vs. Jensen-Shannon on a toy,
bimodal 1-D "teacher" distribution.

The student is restricted to a *single Gaussian* (2 learnable parameters:
mean and log-std) so it genuinely cannot represent the bimodal teacher
exactly - this is what makes the mode-covering vs. mode-seeking difference
visible (with an unrestricted student, every divergence has the same
global optimum: student == teacher).
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.losses import forward_kl, jensen_shannon, reverse_kl  # noqa: E402
from common.plotting import line_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)
N_BINS = 300
X = torch.linspace(-10, 10, N_BINS)


def bimodal_teacher_logits() -> torch.Tensor:
    """A mixture of two well-separated unit-variance Gaussians -> bimodal pmf."""
    pdf = 0.5 * torch.exp(-0.5 * ((X + 4) / 1.0) ** 2) + 0.5 * torch.exp(-0.5 * ((X - 4) / 1.0) ** 2)
    log_pmf = (pdf / pdf.sum() + 1e-12).log()
    return log_pmf.unsqueeze(0)  # [1, N_BINS]; already-normalized log-pmf works fine as "logits"


def gaussian_student_logits(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    std = log_std.exp()
    log_pdf = -0.5 * ((X - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
    log_pmf = log_pdf - torch.logsumexp(log_pdf, dim=-1, keepdim=True)
    return log_pmf.unsqueeze(0)


def fit_student(divergence_fn, teacher_logits, steps: int = 1500, lr: float = 0.1, seed: int = 0):
    torch.manual_seed(seed)
    mu = torch.zeros(1, requires_grad=True)
    log_std = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([mu, log_std], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = divergence_fn(gaussian_student_logits(mu, log_std), teacher_logits)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_pmf = F.softmax(gaussian_student_logits(mu, log_std), dim=-1).squeeze(0)
    return final_pmf, mu.item(), log_std.exp().item()


def entropy(pmf: torch.Tensor) -> float:
    return -(pmf * (pmf + 1e-12).log()).sum().item()


def main():
    os.makedirs(os.path.join(LAB_DIR, "results"), exist_ok=True)
    teacher_logits = bimodal_teacher_logits()
    teacher_pmf = F.softmax(teacher_logits, dim=-1).squeeze(0)

    divergences = {
        "forward_kl": forward_kl,
        "reverse_kl": reverse_kl,
        "jensen_shannon": jensen_shannon,
    }
    labels = {
        "forward_kl": "student fit via forward KL (mode-covering)",
        "reverse_kl": "student fit via reverse KL (mode-seeking)",
        "jensen_shannon": "student fit via Jensen-Shannon (beta=0.5)",
    }

    ys = {"teacher (bimodal)": teacher_pmf.tolist()}
    print(f"Teacher entropy: {entropy(teacher_pmf):.3f} nats\n")
    for key, fn in divergences.items():
        student_pmf, mu, std = fit_student(fn, teacher_logits)
        ys[labels[key]] = student_pmf.tolist()
        print(f"{key:>16}: fitted Gaussian mu={mu:+.2f}, std={std:.2f}, entropy={entropy(student_pmf):.3f} nats")

    line_plot(
        X.tolist(),
        ys,
        xlabel="x",
        ylabel="probability density",
        title="Fitting a single-Gaussian student to a bimodal teacher",
        out_path=os.path.join(LAB_DIR, "results/toy_mode_covering_vs_seeking.png"),
    )
    print("\nSaved results/toy_mode_covering_vs_seeking.png")
    print(
        "\nExpected pattern: forward KL settles near mu~0 with a *wide* "
        "variance to avoid assigning near-zero probability to either mode "
        "(mode-covering - but it wastes mass on the valley between modes, "
        "where the teacher has ~0 density). Reverse KL instead collapses "
        "onto *one* mode with a *narrow* variance, fully ignoring the other "
        "(mode-seeking). Jensen-Shannon (beta=0.5) usually lands in between."
    )


if __name__ == "__main__":
    main()
