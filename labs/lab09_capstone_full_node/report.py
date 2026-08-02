#!/usr/bin/env python
"""Capstone: generate a final markdown + PNG report comparing the teacher,
the non-distilled baseline student, and the distilled student(s) across
the full eval suite from eval_suite.py."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.plotting import bar_plot, grouped_bar_plot  # noqa: E402

LAB_DIR = os.path.dirname(__file__)

MODEL_LABELS = {
    "teacher": "Teacher",
    "baseline_student": "Baseline student\n(no distillation)",
    "sft_student": "SFT-distilled\nstudent",
    "distilled_student": "Final distilled student\n(SFT + on-policy GKD)",
}


def main():
    results_path = os.path.join(LAB_DIR, "results/eval_suite_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found - run eval_suite.py first.")
    results = json.load(open(results_path))

    models = list(results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    metrics = ["gsm8k_pass_at_1", "mmlu_lite_accuracy", "ifeval_lite_accuracy"]
    metric_labels = ["GSM8K pass@1", "MMLU-lite accuracy", "IFEval-lite accuracy"]

    grouped_bar_plot(
        metric_labels,
        {label: [results[model][m] for m in metrics] for model, label in zip(models, labels)},
        ylabel="accuracy",
        title="Capstone eval suite: teacher vs. baseline vs. distilled student",
        out_path=os.path.join(LAB_DIR, "results/report_accuracy.png"),
    )

    bar_plot(
        labels,
        [results[m]["throughput_tok_per_s"] for m in models],
        ylabel="tokens/sec",
        title="Inference throughput",
        out_path=os.path.join(LAB_DIR, "results/report_throughput.png"),
    )
    bar_plot(
        labels,
        [results[m]["peak_mem_gb"] for m in models],
        ylabel="peak GPU memory (GB)",
        title="Peak GPU memory during generation",
        out_path=os.path.join(LAB_DIR, "results/report_memory.png"),
    )

    lines = ["# Distillation Capstone Report", ""]
    lines.append("| Model | GSM8K pass@1 | MMLU-lite | IFEval-lite | Throughput (tok/s) | Peak mem (GB) |")
    lines.append("|---|---|---|---|---|---|")
    for model in models:
        r = results[model]
        lines.append(
            f"| {MODEL_LABELS.get(model, model).replace(chr(10), ' ')} | {r['gsm8k_pass_at_1']:.1%} | "
            f"{r['mmlu_lite_accuracy']:.1%} | {r['ifeval_lite_accuracy']:.1%} | "
            f"{r['throughput_tok_per_s']:.0f} | {r['peak_mem_gb']:.1f} |"
        )
    lines.append("")

    final_student_key = "distilled_student" if "distilled_student" in results else "sft_student"
    if final_student_key in results and "teacher" in results and "baseline_student" in results:
        teacher_acc = results["teacher"]["gsm8k_pass_at_1"]
        baseline_acc = results["baseline_student"]["gsm8k_pass_at_1"]
        final_acc = results[final_student_key]["gsm8k_pass_at_1"]

        lines.append("## Summary")
        lines.append("")
        lines.append(
            f"- Distilled student GSM8K pass@1: **{final_acc:.1%}** vs. teacher **{teacher_acc:.1%}** "
            f"vs. baseline (non-distilled) student **{baseline_acc:.1%}**."
        )
        if teacher_acc != baseline_acc:
            gap_closed = (final_acc - baseline_acc) / (teacher_acc - baseline_acc)
            lines.append(f"- Distillation closed **{gap_closed:.0%}** of the teacher/baseline GSM8K gap.")
        throughput_ratio = results[final_student_key]["throughput_tok_per_s"] / results["teacher"]["throughput_tok_per_s"]
        mem_ratio = results["teacher"]["peak_mem_gb"] / results[final_student_key]["peak_mem_gb"]
        lines.append(
            f"- The distilled student's inference throughput is **{throughput_ratio:.1f}x** the teacher's, "
            f"using **{mem_ratio:.1f}x** less peak GPU memory - the whole point of distillation: "
            "most of the capability, a fraction of the cost."
        )
        lines.append("")

    lines.append("![accuracy comparison](report_accuracy.png)")
    lines.append("")
    lines.append("![throughput comparison](report_throughput.png)")
    lines.append("")
    lines.append("![memory comparison](report_memory.png)")
    lines.append("")

    report_path = os.path.join(LAB_DIR, "results/REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {report_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
