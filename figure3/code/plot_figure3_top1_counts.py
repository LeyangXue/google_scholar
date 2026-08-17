import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator


def period_name(year):
    """Assign a year to the three updated historical periods."""
    if year <= 1979:
        return "Print-dominant era (1950-1979)"
    if year <= 2003:
        return "Digitization and networked-publishing transition (1980-2003)"
    return "Google Scholar era (2004 onward)"


def t_multiplier(n):
    """Return the two-sided 95% t multiplier."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
    except ImportError:
        return 1.96 if n > 1 else np.nan


def load_paper_counts(paper_file):
    """Load paper-level counts calculated for equal proximity halves."""
    columns = [
        "sample",
        "year",
        "n_visibility_references",
        "n_half_references",
        "top1_reference_count",
        "closer_top1_reference_count",
        "distant_top1_reference_count",
        "top1_reference_count_gap",
    ]
    return pd.read_csv(paper_file, usecols=columns)


def load_distribution_bins(bin_file):
    """Load the period-specific count distributions used by the inset."""
    columns = [
        "period",
        "shifted_bin_center",
        "paper_share",
    ]
    bins = pd.read_csv(bin_file, usecols=columns)
    bins["shifted_bin_center"] = pd.to_numeric(
        bins["shifted_bin_center"], errors="coerce"
    )
    bins["paper_share"] = pd.to_numeric(
        bins["paper_share"], errors="coerce"
    )
    return bins.dropna(subset=columns)


def summarize_samples(papers):
    """Average paper-level counts within each year and independent sample."""
    metrics = [
        "top1_reference_count",
        "closer_top1_reference_count",
        "distant_top1_reference_count",
        "top1_reference_count_gap",
    ]
    samples = (
        papers.groupby(["year", "sample"], as_index=False)[metrics]
        .mean()
        .sort_values(["year", "sample"])
    )
    samples["period"] = samples["year"].map(period_name)
    return samples


def summarize_years(samples):
    """Calculate yearly means and 95% CIs across the 30 samples."""
    metrics = [
        "top1_reference_count",
        "closer_top1_reference_count",
        "distant_top1_reference_count",
        "top1_reference_count_gap",
    ]
    rows = []
    for year, group in samples.groupby("year"):
        row = {
            "year": int(year),
            "period": period_name(int(year)),
            "n_samples": group["sample"].nunique(),
        }
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            standard_error = (
                np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan
            )
            half_width = (
                t_multiplier(n) * standard_error if n > 1 else np.nan
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_low"] = mean - half_width
            row[f"{metric}_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def summarize_periods(samples):
    """Calculate period means and 95% CIs across sample-level period means."""
    metrics = [
        "top1_reference_count",
        "closer_top1_reference_count",
        "distant_top1_reference_count",
        "top1_reference_count_gap",
    ]
    sample_periods = (
        samples.groupby(["period", "sample"], as_index=False)[metrics]
        .mean()
    )
    rows = []
    for period, group in sample_periods.groupby("period", sort=False):
        row = {"period": period, "n_samples": group["sample"].nunique()}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            standard_error = (
                np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan
            )
            half_width = (
                t_multiplier(n) * standard_error if n > 1 else np.nan
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_low"] = mean - half_width
            row[f"{metric}_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


def set_plot_style():
    """Apply the same compact journal-style theme as the fraction figure."""
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
    """Mark the digitization transition and Google Scholar periods."""
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
):
    """Draw and label equal-observed-year period means."""
    for start, end in [(1970, 1979), (1980, 2003), (2004, 2020)]:
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
            ax.annotate(
                value_formatter(value),
                ((start + end) / 2, value),
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
    ylabel,
    value_formatter,
):
    """Draw a yearly estimate, its 95% CI, and labeled period means."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    mean = years[f"{metric}_mean"].to_numpy(dtype=float)
    low = years[f"{metric}_low"].to_numpy(dtype=float)
    high = years[f"{metric}_high"].to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.9)
    draw_period_means(ax, years, metric, color, value_formatter)
    finish_axis(ax, "", ylabel)


def draw_count_proximity_panel(ax, years, closer_color, distant_color):
    """Compare mean top-1% counts in the closer and more distant groups."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    series = [
        (
            "closer_top1_reference_count",
            "Semantically closer",
            closer_color,
            -5,
        ),
        (
            "distant_top1_reference_count",
            "Semantically more distant",
            distant_color,
            4,
        ),
    ]
    for metric, label, color, text_offset in series:
        mean = years[f"{metric}_mean"].to_numpy(dtype=float)
        low = years[f"{metric}_low"].to_numpy(dtype=float)
        high = years[f"{metric}_high"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        draw_period_means(
            ax,
            years,
            metric,
            color,
            lambda value: f"{value:.2f}",
            text_offset=text_offset,
        )

    finish_axis(ax, "", "Mean top-1% reference count")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def draw_count_gap_panel(ax, periods, color):
    """Show the distant-minus-closer top-1% count gap by period."""
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
    metric = "top1_reference_count_gap"
    mean = selected[f"{metric}_mean"].to_numpy(dtype=float)
    low = selected[f"{metric}_low"].to_numpy(dtype=float)
    high = selected[f"{metric}_high"].to_numpy(dtype=float)
    x = np.arange(len(period_order))

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
            f"{value:.2f}",
            (position, value),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )

    padding = max(0.08, (high.max() - low.min()) * 0.20)
    ax.set_title("", loc="left", pad=7, fontweight="bold")
    ax.set_ylabel("Mean top-1% count gap\n(distant - closer)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.15, 2.35)
    ax.set_ylim(max(0, low.min() - padding), high.max() + padding)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def draw_count_distribution_inset(ax, distribution_bins, colors):
    """Add period-specific top-1% count probabilities to panel d."""
    inset_ax = ax.inset_axes([0.15, 0.64, 0.36, 0.35], zorder=8)
    period_style = [
        (
            "Print-dominant era (1970-1979)",
            "Print-dominant",
            colors["similarity"],
        ),
        (
            "Digitization transition (1980-2003)",
            "Digitization",
            colors["count"],
        ),
        (
            "Google Scholar era (2004-2020)",
            "Google Scholar",
            colors["closer"],
        ),
    ]

    positive_rows = distribution_bins[
        (distribution_bins["shifted_bin_center"] > 0)
        & (distribution_bins["paper_share"] > 0)
    ]
    if positive_rows.empty:
        raise ValueError("No positive distribution values were found for the inset.")

    for period, label, color in period_style:
        selected = positive_rows[positive_rows["period"] == period].sort_values(
            "shifted_bin_center"
        )
        if selected.empty:
            raise ValueError(f"No distribution data were found for {period}.")
        inset_ax.plot(
            selected["shifted_bin_center"],
            selected["paper_share"],
            color=color,
            linewidth=0.78,
            label=label,
            zorder=3,
        )

    inset_ax.set_xscale("log")
    inset_ax.set_yscale("log")
    inset_ax.set_xlim(
        positive_rows["shifted_bin_center"].min() * 0.90,
        positive_rows["shifted_bin_center"].max() * 1.12,
    )
    inset_ax.set_ylim(
        positive_rows["paper_share"].min() * 0.55,
        positive_rows["paper_share"].max() * 1.65,
    )
    inset_ax.set_xlabel("Top-1% references + 1", fontsize=6, labelpad=0.8)
    inset_ax.set_ylabel("Probability", fontsize=6, labelpad=0.6)
    inset_ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    inset_ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    inset_ax.minorticks_off()
    inset_ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=1.8,
        width=0.5,
        labelsize=6,
        pad=1.0,
    )
    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(0.98)
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4D4D4D")
        spine.set_linewidth(0.48)
    inset_ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=4.5,
        loc="lower left",
        borderaxespad=0.25,
        borderpad=0.15,
        handlelength=1.05,
        handletextpad=0.30,
        labelspacing=0.12,
    )


def add_panel_labels(figure, axes):
    """Place panel labels at consistent figure-level positions."""
    for ax, label in zip(axes.flatten(), ["(a)", "(b)", "(c)", "(d)"]):
        position = ax.get_position()
        figure.text(
            position.x0,
            position.y1 + 0.018,
            label,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def plot_figure(
    semantic_years,
    count_years,
    periods,
    distribution_bins,
    pdf_path,
    png_path,
):
    """Create the 2-by-2 top-1% reference-count Figure 3."""
    set_plot_style()
    colors = {
        "similarity": "#3B6FB6",
        "count": "#D17A22",
        "closer": "#2A8C82",
        "distant": "#7B8085",
        "gap": "#7A5AA6",
    }
    figure, axes = plt.subplots(2, 2, figsize=(6.0, 5.4))

    draw_single_series(
        axes[0, 0],
        semantic_years,
        "mean_similarity_matched",
        colors["similarity"],
        "Mean cosine similarity",
        lambda value: f"{value:.3f}",
    )
    draw_single_series(
        axes[0, 1],
        count_years,
        "top1_reference_count",
        colors["count"],
        "Mean top-1% reference count",
        lambda value: f"{value:.2f}",
    )
    draw_count_proximity_panel(
        axes[1, 0], count_years, colors["closer"], colors["distant"]
    )
    draw_count_gap_panel(axes[1, 1], periods, colors["gap"])
    draw_count_distribution_inset(axes[1, 1], distribution_bins, colors)

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
    add_panel_labels(figure, axes)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=500, bbox_inches="tight")
    #plt.close(figure)


if __name__ == "__main__":
    figure3_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    result_folder = os.path.join(figure3_path, "result", "visibility_top1")
    figure_folder = os.path.join(figure3_path, "figure")

    paper_file = os.path.join(
        result_folder, "figure3_top1_count50_paper_metrics.csv.gz"
    )
    semantic_year_file = os.path.join(
        result_folder, "figure3_top1_year_metrics.csv"
    )
    sample_output_file = os.path.join(
        result_folder, "figure3_top1_count50_sample_metrics.csv"
    )
    year_output_file = os.path.join(
        result_folder, "figure3_top1_count50_year_metrics.csv"
    )
    period_output_file = os.path.join(
        result_folder, "figure3_top1_count50_period_metrics.csv"
    )
    distribution_bin_file = os.path.join(
        result_folder, "figure3_top1_count_distribution_bins.csv"
    )
    pdf_path = os.path.join(figure_folder, "Figure3_top1_counts.pdf")
    png_path = os.path.join(figure_folder, "Figure3_top1_counts.png")

    os.makedirs(figure_folder, exist_ok=True)
    papers = load_paper_counts(paper_file)
    samples = summarize_samples(papers)
    count_years = summarize_years(samples)
    periods = summarize_periods(samples)
    distribution_bins = load_distribution_bins(distribution_bin_file)
    semantic_years = pd.read_csv(semantic_year_file)
    semantic_years = semantic_years[semantic_years["year"] <= 2020].sort_values(
        "year"
    )

    samples.to_csv(sample_output_file, index=False)
    count_years.to_csv(year_output_file, index=False)
    periods.to_csv(period_output_file, index=False)
    plot_figure(
        semantic_years,
        count_years,
        periods,
        distribution_bins,
        pdf_path,
        png_path,
    )
    print("Figure 3 top-1% reference-count plot was saved successfully.")
