import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


def set_plot_style():
    """Apply a compact journal-style plotting theme."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def shade_periods(ax):
    """Mark the updated digitization-transition and Google Scholar periods."""
    ax.axvspan(1980, 2004, color="#D9DDE3", alpha=0.60, linewidth=0, zorder=0)
    ax.axvspan(2004, 2020.5, color="#EFEFEF", alpha=0.78, linewidth=0, zorder=0)


def finish_axis(ax, title, ylabel):
    """Apply shared axis labels and geometry."""
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xlim(1970, 2020)
    ax.set_xticks([1970, 1980, 2004, 2020])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def draw_period_means(
    ax,
    years,
    metric,
    color,
    value_formatter,
    text_offset=4,
    label_x_offsets=None,
):
    """Draw and label equal-observed-year period means."""
    for start, end in [(1970, 1979), (1980, 2003), (2004, 2020)]:
        selected = years[(years["year"] >= start) & (years["year"] <= end)]
        value = selected[f"{metric}_mean"].mean()
        if np.isfinite(value):
            label_x = (start + end) / 2
            if label_x_offsets is not None:
                label_x += label_x_offsets.get((start, end), 0)
            ax.hlines(
                value,
                start,
                end,
                color=color,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                alpha=0.9,
            )
            ax.annotate(
                value_formatter(value),
                (label_x, value),
                xytext=(0, text_offset),
                textcoords="offset points",
                color="#4D4D4D",
                fontsize=7.2,
                ha="center",
                va="bottom" if text_offset >= 0 else "top",
            )


def draw_single_series(
    ax,
    years,
    metric,
    color,
    title,
    ylabel,
    value_formatter,
    percentage=False,
    period_label_x_offsets=None,
):
    """Draw an annual estimate, its 95% CI, and updated period means."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    mean = years[f"{metric}_mean"].to_numpy(dtype=float)
    low = years[f"{metric}_low"].to_numpy(dtype=float)
    high = years[f"{metric}_high"].to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.9)
    draw_period_means(
        ax,
        years,
        metric,
        color,
        value_formatter,
        label_x_offsets=period_label_x_offsets,
    )
    finish_axis(ax, title, ylabel)
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def draw_proximity_panel(ax, years, closest_color, other_color):
    """Compare top-1% visibility in the closest quartile and other references."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    series = [
        (
            "closest_high_visibility_share",
            "Semantically closer",
            closest_color,
        ),
        (
            "other_high_visibility_share",
            "Semantically more distant",
            other_color,
        ),
    ]
    for metric, label, color in series:
        mean = years[f"{metric}_mean"].to_numpy(dtype=float)
        low = years[f"{metric}_low"].to_numpy(dtype=float)
        high = years[f"{metric}_high"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        text_offset = -5 if metric == "closest_high_visibility_share" else 4
        draw_period_means(
            ax,
            years,
            metric,
            color,
            lambda value: f"{value:.1%}",
            text_offset=text_offset,
        )

    finish_axis(
        ax,
        "(c)",
        "Top-1% reference fraction",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def draw_gap_panel(ax, periods, color):
    """Show the top-1% visibility gap between proximity groups by period."""
    period_order = [
        "Print-dominant era (1950-1979)",
        "Digitization and networked-publishing transition (1980-2003)",
        "Google Scholar era (2004 onward)",
    ]
    labels = [
        "Print-dominant\n1950-1979",
        "Digitization transition\n1980-2003",
        "Google Scholar\n2004-2020",
    ]
    selected = periods.set_index("period").loc[period_order]
    mean = selected["high_visibility_gap_mean"].to_numpy(dtype=float)
    low = selected["high_visibility_gap_low"].to_numpy(dtype=float)
    high = selected["high_visibility_gap_high"].to_numpy(dtype=float)
    x = np.arange(len(period_order))

    ax.axhline(0, color="#7B8085", linewidth=0.8, zorder=0)
    ax.plot(x, mean, color=color, linewidth=1.3, zorder=2)
    ax.errorbar(
        x,
        mean,
        yerr=[mean - low, high - mean],
        fmt="o",
        color=color,
        markersize=5,
        linewidth=1.3,
        capsize=3,
        zorder=3,
    )
    for position, value in zip(x, mean):
        ax.annotate(
            f"{value:.1%}",
            (position, value),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_title("(d)", loc="left", pad=7, fontweight="bold")
    ax.set_ylabel("Top-1% fraction gap\n(distant - closer)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.15, 2.35)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(0.038, 0.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def plot_figure(years, periods, pdf_path, png_path):
    """Create the 2-by-2 top-1% visibility Figure 3."""
    set_plot_style()
    colors = {
        "similarity": "#3B6FB6",
        "visibility": "#D17A22",
        "closest": "#2A8C82",
        "other": "#7B8085",
        "gap": "#7A5AA6",
    }
    figure, axes = plt.subplots(2, 2, figsize=(6, 5.4))

    draw_single_series(
        axes[0, 0],
        years,
        "mean_similarity_matched",
        colors["similarity"],
        "(a)",
        "Mean cosine similarity",
        lambda value: f"{value:.3f}",
    )
    draw_single_series(
        axes[0, 1],
        years,
        "high_visibility_share",
        colors["visibility"],
        "(b)",
        "Top-1% reference fraction",
        lambda value: f"{value:.1%}",
        percentage=True,
        period_label_x_offsets={(1980, 2003): -4.0},
    )
    draw_proximity_panel(axes[1, 0], years, colors["closest"], colors["other"])
    draw_gap_panel(axes[1, 1], periods, colors["gap"])

    period_handles = [
        Line2D(
            [0], [0], color="#FFFFFF", marker="s", markeredgecolor="#BDBDBD",
            markersize=7, linewidth=0,
            label="Print-dominant era (1950-1979)",
        ),
        Line2D(
            [0], [0], color="#D9DDE3", linewidth=6, alpha=0.90,
            label="Digitization and networked-publishing transition (1980-2003)",
        ),
        Line2D(
            [0], [0], color="#EFEFEF", linewidth=6, alpha=0.95,
            label="Google Scholar era (2004 onward)",
        ),
    ]
    figure.legend(
        handles=period_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        fontsize=7.0,
        handlelength=1.4,
        columnspacing=0.9,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.95,
        bottom=0.19,
        wspace=0.30,
        hspace=0.42,
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=500, bbox_inches="tight")
    #plt.close(figure)


if __name__ == "__main__":
    
    figure3_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    result_folder = os.path.join(figure3_path, "result", "visibility_top1")
    year_file = os.path.join(result_folder, "figure3_top1_year_metrics.csv")
    period_file = os.path.join(result_folder, "figure3_top1_period_sample_ci.csv")
    output_folder = os.path.join(figure3_path, "figure")
    pdf_path = os.path.join(output_folder, "Figure3_top1_visibility.pdf")
    png_path = os.path.join(output_folder, "Figure3_top1_visibility.png")

    os.makedirs(output_folder, exist_ok=True)
    years = pd.read_csv(year_file)
    years = years[years["year"] <= 2020].sort_values("year")
    periods = pd.read_csv(period_file)
    plot_figure(years, periods, pdf_path, png_path)
    print("Figure 3 top-1% visibility plot was saved successfully.")
