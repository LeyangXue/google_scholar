#!/usr/bin/env python3
"""SI-2 figure: rank mobility of the citation elite (supports Figure 2).

Visual conventions match Figure 2 (figure2/code/plot_elite_persistence.py):
  - dataset colors: MAG #3C5488, OpenAlex #E64B35 (Spearman lines / period means)
  - period colors:  Print #777777, Digitization #00A087, Google Scholar #E64B35
  - background era shading: 1980-2004 #D9DDE3, 2004+ #EFEFEF
  - period means drawn as horizontal segments with 95% CI bands and value labels

Layout (2 rows x 3 columns), matching Figure 2:
    row 1 MAG:      (a) Spearman 2->4   (b) Spearman 5->10   (c) rank persistence
    row 2 OpenAlex: (d) Spearman 2->4   (e) Spearman 5->10   (f) rank persistence
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULT = "/Volumes/lydisk/work/work11/google_scholar/figureS2/result"
OUTDIR = "/Volumes/lydisk/work/work11/google_scholar/figureS2/figure"

DATASET_COLORS = {"MAG": "#3C5488", "OpenAlex": "#E64B35"}
PERIOD_ORDER = [
    "Print-dominant era (1950–1979)",
    "Digitization and networked-publishing transition (1980–2003)",
    "Google Scholar era (2004 onward)",
]
PERIOD_COLORS = {
    "Print-dominant era (1950–1979)": "#777777",
    "Digitization and networked-publishing transition (1980–2003)": "#00A087",
    "Google Scholar era (2004 onward)": "#E64B35",
}
PERIOD_SHORT = {
    "Print-dominant era (1950–1979)": "Print-dominant (1950–1979)",
    "Digitization and networked-publishing transition (1980–2003)":
        "Digitization transition (1980–2003)",
    "Google Scholar era (2004 onward)": "Google Scholar era (2004 onward)",
}
BINS = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%"]


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)


def add_historical_periods(ax, end_year=2020):
    ax.axvspan(1980, 2004, color="#D9DDE3", alpha=0.60, linewidth=0, zorder=0)
    ax.axvspan(2004, end_year, color="#EFEFEF", alpha=0.78, linewidth=0, zorder=0)


def plot_spearman(ax, yearly, period, dataset, window, show_ylabel, y_range):
    color = DATASET_COLORS[dataset]
    annual = yearly[(yearly["Dataset"] == dataset)
                    & (yearly["Window"] == window)].sort_values("Publication_year")
    pstat = period[(period["Dataset"] == dataset) & (period["Window"] == window)]

    add_historical_periods(ax)
    ax.plot(annual["Publication_year"], annual["Spearman"], color=color,
            linewidth=1.8, solid_capstyle="round", zorder=3)

    for _, row in pstat.iterrows():
        s, e = row["Start_year"], row["End_year"]
        mean, lo, hi = row["Mean_spearman"], row["CI_lower"], row["CI_upper"]
        ax.fill_between([s, e], [lo, lo], [hi, hi], color=color, alpha=0.11,
                        linewidth=0, zorder=1)
        ax.hlines(mean, s, e, color=color, linewidth=3.0, alpha=0.76, zorder=2)
        ax.text((s + e) / 2, hi + 0.012, f"{mean:.2f}", color="#4D4D4D",
                fontsize=7.5, ha="center", va="bottom", zorder=5)

    ax.set_xlim(1950, 2020)
    ax.set_xticks([1960, 1980, 2000, 2020])
    ax.set_ylim(*y_range)
    ax.set_xlabel("Publication year", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Top-1% rank correlation\n(Spearman, age a → 2a)", fontsize=9)
    style_axis(ax)


def plot_persistence(ax, period_df, dataset, show_ylabel, show_legend):
    d = period_df[(period_df["Dataset"] == dataset)
                  & (period_df["Window"] == "5yr")]
    x = np.arange(len(BINS))
    width = 0.26
    for pi, per in enumerate(PERIOD_ORDER):
        sub = d[d["Period"] == per].set_index("Bin")
        vals = np.array([sub.loc[b, "Stay_prob"] if b in sub.index else np.nan
                         for b in BINS])
        lo = np.array([sub.loc[b, "Stay_CI_lower"] if b in sub.index else np.nan
                       for b in BINS])
        hi = np.array([sub.loc[b, "Stay_CI_upper"] if b in sub.index else np.nan
                       for b in BINS])
        xpos = x + (pi - 1) * width
        ax.bar(xpos, vals, width, color=PERIOD_COLORS[per], edgecolor="white",
               linewidth=0.4, zorder=3,
               label=PERIOD_SHORT[per] if show_legend else None)
        ax.errorbar(xpos, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor="#404040", elinewidth=0.7, capsize=1.6, zorder=4)
        for xi, v, h in zip(xpos, vals, hi):
            if np.isfinite(v):
                ax.text(xi, h + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.6, color="#4D4D4D", rotation=90, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(BINS, fontsize=8)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Citation-rank bin at age 5", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Prob. of staying in the\nsame 1% rank bin", fontsize=9)
    style_axis(ax)
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.03, 1.05), frameon=False,
                  fontsize=6.6, handlelength=1.1, borderaxespad=0.0)


def add_panel_labels(fig, axes, labels):
    for ax, label in zip(axes, labels):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.045, pos.y1 + 0.012, label, fontsize=11,
                 fontweight="bold", ha="left", va="bottom")


def add_period_legend(fig):
    handles = [
        Patch(facecolor="#FFFFFF", edgecolor="#BDBDBD", linewidth=0.6),
        Patch(facecolor="#D9DDE3", alpha=0.90, edgecolor="none"),
        Patch(facecolor="#EFEFEF", alpha=0.95, edgecolor="none"),
        Line2D([0], [0], color="#666666", linewidth=3.0, alpha=0.76),
    ]
    labels = [
        "Print-dominant era (1950–1979)",
        "Digitization and networked-publishing transition (1980–2003)",
        "Google Scholar era (2004 onward)",
        "Period mean (95% CI shaded)",
    ]
    fig.legend(handles[:3], labels[:3], loc="lower center",
               bbox_to_anchor=(0.50, 0.028), ncol=3, frameon=False,
               fontsize=7.1, handlelength=1.5, columnspacing=1.0)
    fig.legend(handles[3:], labels[3:], loc="lower center",
               bbox_to_anchor=(0.50, 0.004), ncol=1, frameon=False,
               fontsize=7.1, handlelength=1.8)


def main():
    plt.rcParams.update({
        "font.family": "Arial", "axes.labelcolor": "#262626",
        "xtick.color": "#262626", "ytick.color": "#262626",
        "text.color": "#262626", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    yearly = pd.read_csv(os.path.join(RESULT, "spearman_topk_yearly.csv"))
    period = pd.read_csv(os.path.join(RESULT, "spearman_topk_period.csv"))
    trans = pd.read_csv(os.path.join(RESULT, "transition_stay_period.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
    y_range = (0.66, 0.98)
    for r, ds in enumerate(["MAG", "OpenAlex"]):
        plot_spearman(axes[r, 0], yearly, period, ds, "2yr", True, y_range)
        plot_spearman(axes[r, 1], yearly, period, ds, "5yr", False, y_range)
        plot_persistence(axes[r, 2], trans, ds, True, show_legend=(r == 0))

    column_titles = [
        "Rank stability (ages 2→4)",
        "Rank stability (ages 5→10)",
        "Rank mobility (top 5%)",
    ]
    for c, t in enumerate(column_titles):
        axes[0, c].set_title(t, fontsize=10.2, pad=10)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.155,
                        wspace=0.30, hspace=0.33)
    add_panel_labels(fig, axes.flatten(),
                     ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])
    for r, ds in enumerate(["MAG", "OpenAlex"]):
        pos = axes[r, 0].get_position()
        fig.text(0.028, (pos.y0 + pos.y1) / 2, ds, rotation=90, ha="center",
                 va="center", fontsize=11, fontweight="bold")
    add_period_legend(fig)

    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "FigureS2_rank_mobility.png"), dpi=600)
    fig.savefig(os.path.join(OUTDIR, "FigureS2_rank_mobility.pdf"))
    plt.close(fig)
    print("saved figure to", OUTDIR)


if __name__ == "__main__":
    main()
