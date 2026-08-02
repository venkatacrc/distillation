"""Matplotlib helpers used across labs so every plot looks consistent and
saves straight to a PNG (no interactive backend needed on a headless node).
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGSIZE = (7, 4.5)
DPI = 150


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def line_plot(
    x,
    ys: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    xscale: str = "linear",
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, y in ys.items():
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    if len(ys) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def bar_plot(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    out_path: str,
    value_labels: bool = True,
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if value_labels:
        for bar, v in zip(bars, values):
            ax.annotate(
                f"{v:.3f}" if isinstance(v, float) else str(v),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def grouped_bar_plot(
    categories: list[str],
    group_values: dict[str, list[float]],
    ylabel: str,
    title: str,
    out_path: str,
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_groups = len(group_values)
    width = 0.8 / max(1, n_groups)
    x = range(len(categories))
    for i, (label, values) in enumerate(group_values.items()):
        offsets = [xi + (i - (n_groups - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def scatter_plot(
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    labels: Optional[list[str]] = None,
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(x, y)
    if labels:
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(li, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path
