"""
Paper figures, drawn from results/ into figures/ (PDF for the paper, PNG to look at):
  qaoa_depth    QAOA P(opt) against depth, mean over fresh instances with a 95% t-interval
                band (grid.py's eval rows at the chosen weights); XY mixer n = 3-5, X mixer n = 3-4
  sa_tightness  annealing P(opt) against window half-width (experiments.py tightness)
  resources     qubits and two-qubit terms per QAOA layer (experiments.resource_counts)
Colours are the dataviz reference palette's first three slots plus a gray for the E0
control, validated for colour-blind and normal-vision separation on white; every encoding
also has its own marker, so the figures survive greyscale print.
Run:  python figures.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import grid
from experiments import ENCODINGS, SIZES, WIDTHS, resource_counts

STYLE = {"E0": ("#6e6d69", "o"), "E2": ("#1baf7a", "^"),
         "E1": ("#2a78d6", "s"), "E1+C": ("#eb6834", "D")}          # drawing and legend order
LABEL = {"E0": "E0  no window term", "E2": "E2  Salehi et al.",
         "E1": "E1  ours", "E1+C": "E1+C  ours, look-ahead"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "lines.markeredgewidth": 0.8,
    "lines.markeredgecolor": "white", "legend.frameon": False, "legend.fontsize": 7,
    "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def series(ax, x, groups, enc):
    """Mean line with markers over a 95% t-interval band; groups[i] = per-instance values at x[i]."""
    color, marker = STYLE[enc]
    mean = np.array([np.mean(g) for g in groups])
    half = np.array([grid.ci(g) for g in groups])
    ax.fill_between(x, np.clip(mean - half, 0, 1), np.clip(mean + half, 0, 1), color=color, alpha=0.12, lw=0)
    ax.plot(x, mean, color=color, marker=marker, label=LABEL[enc])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}")
    plt.close(fig)


def qaoa_depth():
    rows = grid.load()
    best = grid.best_weights(rows)
    vals = defaultdict(list)                              # (mixer, n, enc, p) → P(opt) per instance
    for r in rows:
        if r["stage"] == "eval" and (r["p1"], r["p2"]) == best[r["mixer"], r["n"], r["enc"]]:
            vals[r["mixer"], r["n"], r["enc"], r["p"]].append(r["p_opt"])
    depths = range(1, grid.DEPTH + 1)
    fig, axes = plt.subplots(2, 3, figsize=(4.8, 3.3))
    for ax, (mixer, n) in zip(axes.flat, grid.CASES):
        for enc in (e for e in STYLE if e in grid.ENCODINGS):
            series(ax, depths, [vals[mixer, n, enc, p] for p in depths], enc)
        ax.set_title(f"{mixer.upper()} mixer, n = {n}")
        ax.set_xticks([1, 4, 8, 12])
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(opt)")
    for ax in (*axes[1, :2], axes[0, 2]):
        ax.set_xlabel("depth p")
    axes[1, 2].axis("off")
    axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="center",
                      title="mean over instances,\nband: 95% t-interval", title_fontsize=7)
    fig.tight_layout(h_pad=1.2)
    save(fig, "qaoa_depth")


def sa_tightness():
    vals = defaultdict(list)                              # (n, width, enc) → P(opt) per instance
    for r in csv.DictReader(open("results/sa_tightness.csv")):
        if r["phase"] == "eval":
            vals[int(r["n"]), int(r["width"]), r["encoding"]].append(float(r["p_opt"]))
    x = range(len(WIDTHS))
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.9), sharey=True)
    for ax, n in zip(axes, SIZES):
        for enc in STYLE:
            series(ax, x, [vals[n, w, enc] for w in WIDTHS], enc)
        ax.set_title(f"n = {n}")
        ax.set_xticks(x, WIDTHS)
        ax.set_xlabel("window half-width")
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("P(opt)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, "sa_tightness")


def resources():
    counts = resource_counts()
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.9))
    for ax, (k, title) in zip(axes, ((0, "qubits"), (1, "two-qubit terms per layer, X mixer"))):
        for i, enc in enumerate(STYLE):
            color, marker = STYLE[enc]
            x = np.arange(len(SIZES)) + (i - 1.5) * 0.13         # side by side, so equal counts stay visible
            ax.plot(x, [counts[n, enc][k] for n in SIZES], ls="none", color=color, marker=marker, ms=6,
                    label=LABEL[enc])
            if enc in ("E1", "E2"):                               # label the n = 5 contrast only
                v = counts[SIZES[-1], enc][k]
                ax.annotate(f"{v:,.0f}", (x[-1], v), textcoords="offset points", color=INK2, fontsize=6.5,
                            **(dict(xytext=(-6, 0), ha="right", va="center") if enc == "E2"
                               else dict(xytext=(0, -11), ha="center")))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xticks(range(len(SIZES)), [f"n = {n}" for n in SIZES])
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, "resources")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    qaoa_depth()
    sa_tightness()
    resources()
