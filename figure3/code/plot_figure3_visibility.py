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
    """Mark the digital-publishing and Google Scholar periods."""
    ax.axvspan(1990, 2004, color="#8F969C", alpha=0.18, linewidth=0, zorder=0)
    ax.axvspan(2004, 2020.5, color="#D7D9DB", alpha=0.34, linewidth=0, zorder=0)
    ax.axvline(2004, color="#71777C", linewidth=0.8, zorder=1)


def finish_axis(ax, title, ylabel):
    """Apply shared axis labels and geometry."""
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xlim(1970, 2020)
    ax.set_xticks([1970, 1990, 2004, 2020])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def draw_period_means(ax, years, metric, color):
    """Draw equal-year period means as short dashed segments."""
    for start, end in [(1970, 1989), (1990, 2003), (2004, 2020)]:
        selected = years[(years["year"] >= start) & (years["year"] <= end)]
        value = selected[f"{metric}_mean"].mean()
        if np.isfinite(value):
            ax.hlines(
                value,
                start,
                end,
                color=color,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                alpha=0.9,
            )


def draw_single_series(ax, years, metric, color, title, ylabel, percentage=False):
    """Draw an annual estimate, its 95 percent CI, and period means."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    mean = years[f"{metric}_mean"].to_numpy(dtype=float)
    low = years[f"{metric}_low"].to_numpy(dtype=float)
    high = years[f"{metric}_high"].to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.9)
    draw_period_means(ax, years, metric, color)
    finish_axis(ax, title, ylabel)
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def draw_closest_panel(ax, years, closest_color, other_color):
    """Compare high visibility in the closest quartile and remaining references."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    series = [
        ("closest_high_visibility_share", "Closest 25%", closest_color),
        ("other_high_visibility_share", "Remaining 75%", other_color),
    ]
    for metric, label, color in series:
        mean = years[f"{metric}_mean"].to_numpy(dtype=float)
        low = years[f"{metric}_low"].to_numpy(dtype=float)
        high = years[f"{metric}_high"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)

    finish_axis(
        ax,
        "(c) High visibility by semantic proximity",
        "High-visibility reference share",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def draw_gap_panel(ax, periods, color):
    """Show the less-similar minus closest high-visibility gap by period."""
    period_order = [
        "Pre-digital (1970-1989)",
        "Digital publishing (1990-2003)",
        "Google Scholar era (2004 onward)",
    ]
    labels = [
        "Pre-digital\n1970-1989",
        "Digital\n1990-2003",
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
            xytext=(7, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("(d) High-visibility gap by period", loc="left", pad=7, fontweight="bold")
    ax.set_ylabel("Remaining 75% - closest 25%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.15, 2.35)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    padding = max(0.004, (high.max() - low.min()) * 0.30)
    ax.set_ylim(max(0, low.min() - padding), high.max() + padding)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def plot_figure(years, periods, pdf_path, png_path):
    """Create the 2 by 2 Figure 3."""
    set_plot_style()
    colors = {
        "distance": "#3B6FB6",
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
        colors["distance"],
        "(a) Mean semantic similarity",
        "Mean cosine similarity",
    )
    draw_single_series(
        axes[0, 1],
        years,
        "high_visibility_share",
        colors["visibility"],
        "(b) High-visibility references",
        "High-visibility reference share",
        percentage=True,
    )
    draw_closest_panel(axes[1, 0], years, colors["closest"], colors["other"])
    draw_gap_panel(axes[1, 1], periods, colors["gap"])

    period_handles = [
        Line2D(
            [0], [0], color="#8F969C", linewidth=6, alpha=0.25,
            label="Digital publishing (1990-2003)",
        ),
        Line2D(
            [0], [0], color="#D7D9DB", linewidth=6, alpha=0.65,
            label="Google Scholar era (2004 onward)",
        ),
    ]
    figure.legend(
        handles=period_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        fontsize=8,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.95,
        bottom=0.18,
        wspace=0.30,
        hspace=0.42,
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=400, bbox_inches="tight")
    #plt.close(figure)


if __name__ == "__main__":
    figure3_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    result_file = os.path.join(
        figure3_path,
        "result",
        "visibility",
        "figure3_visibility_year_metrics.csv",
    )
    period_file = os.path.join(
        figure3_path,
        "result",
        "visibility",
        "figure3_visibility_period_metrics.csv",
    )
    output_folder = os.path.join(figure3_path, "figure")
    pdf_path = os.path.join(output_folder, "Figure3_citation_diversity.pdf")
    png_path = os.path.join(output_folder, "Figure3_citation_diversity.png")

    os.makedirs(output_folder, exist_ok=True)
    years = pd.read_csv(result_file)
    years = years[years["year"] <= 2020].sort_values("year")
    periods = pd.read_csv(period_file)
    plot_figure(years, periods, pdf_path, png_path)
    print("Figure 3 was saved successfully.")
