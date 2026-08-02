#!/usr/bin/env python
"""Lab01 step 4: visualize the temperature/alpha sweep and compare teacher
vs. baseline student vs. best distilled student."""
from __future__ import annotations

import json
import os
import sys

LAB_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(LAB_DIR, "..", "..")))
from common.plotting import bar_plot, line_plot  # noqa: E402


def main():
    results_dir = os.path.join(LAB_DIR, "results")
    teacher = json.load(open(os.path.join(results_dir, "teacher_metrics.json")))
    baseline = json.load(open(os.path.join(results_dir, "baseline_metrics.json")))
    sweep = json.load(open(os.path.join(results_dir, "kd_sweep.json")))

    alphas = sorted({r["alpha"] for r in sweep if r["alpha"] > 0})
    temperatures = sorted({r["temperature"] for r in sweep})

    ys = {}
    for alpha in alphas:
        accs = []
        for t in temperatures:
            match = [r for r in sweep if r["alpha"] == alpha and r["temperature"] == t]
            accs.append(match[0]["val_accuracy"] if match else float("nan"))
        ys[f"alpha={alpha}"] = accs

    line_plot(
        temperatures,
        ys,
        xlabel="Temperature T",
        ylabel="Validation accuracy",
        title="SST-2 KD student accuracy vs. temperature",
        out_path=os.path.join(results_dir, "accuracy_vs_temperature.png"),
    )

    alpha0_matches = [r for r in sweep if r["alpha"] == 0.0]
    best_kd = max(sweep, key=lambda r: r["val_accuracy"])

    bar_plot(
        ["Teacher\n(BERT-base)", "Student baseline\n(hard labels only)", f"Student + KD\n(T={best_kd['temperature']}, a={best_kd['alpha']})"],
        [teacher["val_accuracy"], baseline["val_accuracy"], best_kd["val_accuracy"]],
        ylabel="Validation accuracy",
        title="Teacher vs. baseline student vs. best KD student",
        out_path=os.path.join(results_dir, "teacher_vs_student.png"),
    )

    print(f"Teacher accuracy:        {teacher['val_accuracy']:.4f}")
    if alpha0_matches:
        print(f"Baseline student (hard): {baseline['val_accuracy']:.4f}  (alpha=0 sweep point: {alpha0_matches[0]['val_accuracy']:.4f})")
    print(f"Best KD student:         {best_kd['val_accuracy']:.4f}  at T={best_kd['temperature']}, alpha={best_kd['alpha']}")
    print(f"Saved plots to {results_dir}/accuracy_vs_temperature.png and {results_dir}/teacher_vs_student.png")


if __name__ == "__main__":
    main()
