#!/usr/bin/env python3
"""FigureS4: citation-semantics trend is robust to the distance measure (1x3).

Three panels, one per measure (mass-center distance, Jaccard similarity, cosine
similarity), each showing the yearly mean over sampled citations with the three
historical-period means and moving-block bootstrap 95% CIs. Style matches
Figure 2 / Figure S2 (era shading, period-mean segments with value labels).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULT = "/Volumes/lydisk/work/work11/google_scholar/figureS4/result"
OUTDIR = "/Volumes/lydisk/work/work11/google_scholar/figureS4/figure"

METHOD_ORDER = ["masscenter", "jaccard", "embeding"]
PANEL_TITLE = {
    "masscenter": "Mass-center distance",
    "jaccard": "Jaccard similarity",
    "embeding": "Cosine similarity",
}
YLABEL = {
    "masscenter": "Mean distance\n(higher = more distant)",
    "jaccard": "Mean Jaccard similarity\n(higher = more similar)",
    "embeding": "Mean cosine similarity\n(higher = more similar)",
}
LINE_COLOR = {"masscenter": "#3C5488", "jaccard": "#00A087", "embeding": "#E64B35"}
PERIOD_ORDER = [
    "Print-dominant era (1950–1979)",
    "Digitization and networked-publishing transition (1980–2003)",
    "Google Scholar era (2004 onward)",
]


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)


def add_eras(ax):
    ax.axvspan(1980, 2004, color="#D9DDE3", alpha=0.60, linewidth=0, zorder=0)
    ax.axvspan(2004, 2020, color="#EFEFEF", alpha=0.78, linewidth=0, zorder=0)


def plot_method(ax, yearly, period, method):
    color = LINE_COLOR[method]
    yy = yearly[yearly["method"] == method].sort_values("year")
    pp = period[period["method"] == method]
    add_eras(ax)
    ax.plot(yy["year"], yy["mean"], color=color, linewidth=1.8,
            solid_capstyle="round", zorder=3)

    # decide label offset direction from data span
    span = yy["mean"].max() - yy["mean"].min()
    off = 0.045 * span
    for _, r in pp.iterrows():
        s, e, mean = r["start_year"], r["end_year"], r["mean"]
        lo, hi = r["ci_lower"], r["ci_upper"]
        ax.fill_between([s, e], [lo, lo], [hi, hi], color=color, alpha=0.12,
                        linewidth=0, zorder=1)
        ax.hlines(mean, s, e, color=color, linewidth=3.0, alpha=0.78, zorder=2)
        ax.text((s + e) / 2, hi + off, f"{mean:.3f}", color="#4D4D4D",
                fontsize=7.2, ha="center", va="bottom", zorder=5)

    ax.set_xlim(1950, 2020)
    ax.set_xticks([1960, 1980, 2000, 2020])
    lo_all = min(yy["mean"].min(), pp["ci_lower"].min())
    hi_all = max(yy["mean"].max(), pp["ci_upper"].max())
    pad = 0.14 * (hi_all - lo_all)
    ax.set_ylim(lo_all - pad, hi_all + pad * 1.7)
    ax.set_title(PANEL_TITLE[method], fontsize=10.2, pad=8)
    ax.set_xlabel("Publication year", fontsize=9)
    ax.set_ylabel(YLABEL[method], fontsize=8.8)
    style_axis(ax)


def add_period_legend(fig):
    handles = [
        Patch(facecolor="#FFFFFF", edgecolor="#BDBDBD", linewidth=0.6),
        Patch(facecolor="#D9DDE3", alpha=0.90, edgecolor="none"),
        Patch(facecolor="#EFEFEF", alpha=0.95, edgecolor="none"),
        Line2D([0], [0], color="#666666", linewidth=3.0, alpha=0.78),
    ]
    labels = [
        "Print-dominant era (1950–1979)",
        "Digitization and networked-publishing transition (1980–2003)",
        "Google Scholar era (2004 onward)",
        "Period mean (95% CI shaded)",
    ]
    fig.legend(handles[:3], labels[:3], loc="lower center",
               bbox_to_anchor=(0.50, 0.045), ncol=3, frameon=False,
               fontsize=7.2, handlelength=1.5, columnspacing=1.0)
    fig.legend(handles[3:], labels[3:], loc="lower center",
               bbox_to_anchor=(0.50, 0.012), ncol=1, frameon=False,
               fontsize=7.2, handlelength=1.8)


def main():
    plt.rcParams.update({
        "font.family": "Arial", "axes.labelcolor": "#262626",
        "xtick.color": "#262626", "ytick.color": "#262626",
        "text.color": "#262626", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    yearly = pd.read_csv(os.path.join(RESULT, "semantic_methods_year.csv"))
    period = pd.read_csv(os.path.join(RESULT, "semantic_methods_period.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5))
    for ax, method in zip(axes, METHOD_ORDER):
        plot_method(ax, yearly, period, method)

    fig.subplots_adjust(left=0.085, right=0.988, top=0.88, bottom=0.30,
                        wspace=0.42)
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.045, pos.y1 + 0.02, lab, fontsize=11,
                 fontweight="bold", ha="left", va="bottom")
    add_period_legend(fig)

    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "FigureS4_semantic_methods.png"), dpi=600)
    fig.savefig(os.path.join(OUTDIR, "FigureS4_semantic_methods.pdf"))
    plt.close(fig)
    print("saved figure to", OUTDIR)


if __name__ == "__main__":
    main()
