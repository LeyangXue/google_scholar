import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def period_name(year):
    """Assign a citing year to one of the three historical periods."""
    if year <= 1979:
        return "Print-dominant era (1970-1979)"
    if year <= 2003:
        return "Digitization transition (1980-2003)"
    return "Google Scholar era (2004-2020)"


def load_paper_counts(paper_file, start_year, end_year):
    """Load one top-1% reference count for each unique citing paper and year."""
    columns = [
        "CitingPublicationId",
        "year",
        "top1_reference_count",
    ]
    papers = pd.read_csv(paper_file, usecols=columns)
    papers = papers.dropna(subset=columns).copy()
    papers["year"] = pd.to_numeric(papers["year"], errors="coerce")
    papers["top1_reference_count"] = pd.to_numeric(
        papers["top1_reference_count"], errors="coerce"
    )
    papers = papers.dropna(subset=["year", "top1_reference_count"])
    papers = papers[
        (papers["year"] >= start_year)
        & (papers["year"] <= end_year)
        & (papers["top1_reference_count"] >= 0)
    ].copy()

    keys = ["year", "CitingPublicationId"]
    conflicting = (
        papers.groupby(keys)["top1_reference_count"]
        .nunique()
        .gt(1)
        .sum()
    )
    if conflicting:
        raise ValueError(
            f"Found {conflicting} paper-year records with inconsistent counts."
        )

    papers = papers.drop_duplicates(keys, keep="first")
    papers["year"] = papers["year"].astype(int)
    papers["period"] = papers["year"].map(period_name)
    return papers


def calculate_distributions(papers, period_order, bins):
    """Calculate normalized distributions using shared logarithmic bins."""
    pooled_counts = papers["top1_reference_count"].to_numpy(dtype=float)
    maximum_shifted_count = np.max(pooled_counts) + 1.0
    bin_edges = np.logspace(
        np.log10(1.0),
        np.log10(maximum_shifted_count),
        bins + 1,
    )
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    distributions = {}
    summary_rows = []
    bin_rows = []

    for period in period_order:
        values = papers.loc[
            papers["period"] == period, "top1_reference_count"
        ].to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"No papers were found for {period}.")

        shifted_values = values + 1.0
        counts, _ = np.histogram(shifted_values, bins=bin_edges)
        probabilities = counts / counts.sum()
        probability_density = probabilities / bin_widths
        distributions[period] = probability_density

        summary_rows.append(
            {
                "period": period,
                "n_papers": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "zero_share": np.mean(values == 0),
                "p90": np.quantile(values, 0.90),
                "p95": np.quantile(values, 0.95),
                "p99": np.quantile(values, 0.99),
                "maximum": np.max(values),
            }
        )

        for index, probability in enumerate(probabilities):
            bin_rows.append(
                {
                    "period": period,
                    "bin_index": index + 1,
                    "shifted_bin_left": bin_edges[index],
                    "shifted_bin_right": bin_edges[index + 1],
                    "shifted_bin_center": bin_centers[index],
                    "count_bin_left": bin_edges[index] - 1.0,
                    "count_bin_right": bin_edges[index + 1] - 1.0,
                    "paper_share": probability,
                    "probability_density": probability_density[index],
                }
            )

    summary = pd.DataFrame(summary_rows)
    bin_data = pd.DataFrame(bin_rows)
    return distributions, bin_edges, bin_centers, summary, bin_data


def set_plot_style():
    """Apply a compact journal-style theme consistent with the main figure."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_distribution(
    distributions,
    bin_edges,
    bin_centers,
    summary,
    period_order,
    colors,
    log_y,
    pdf_path,
    png_path,
):
    """Plot point-line distributions on shared logarithmic count bins."""
    set_plot_style()
    figure, ax = plt.subplots(figsize=(3.5, 2.8))
    summary_index = summary.set_index("period")

    for period in period_order:
        mean_value = summary_index.loc[period, "mean"]
        label = f"{period}\nMean = {mean_value:.2f}"
        density = distributions[period]
        observed = density > 0
        ax.plot(
            bin_centers[observed],
            density[observed],
            color=colors[period],
            linewidth=1.35,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=colors[period],
            markeredgewidth=0.95,
            markersize=3.2,
            label=label,
            zorder=3,
        )

    ax.set_xlabel(
        "Top-1% reference count per citing paper + 1\n"
        "(1 represents zero)"
    )
    ax.set_ylabel("Probability density")
    ax.set_xscale("log")
    ax.set_xlim(bin_edges[0] * 0.93, bin_edges[-1] * 1.04)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Probability density")
    else:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#222222")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=3.5,
        width=0.75,
        top=True,
        right=True,
        pad=2.5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=2.0,
        width=0.6,
        top=True,
        right=True,
    )
    ax.tick_params(labeltop=False, labelright=False)
    ax.legend(
        frameon=False,
        fontsize=6.8,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        borderaxespad=0.0,
        handlelength=2.2,
        handletextpad=0.65,
        labelspacing=0.55,
    )
    
    figure.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.16)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    
    figure3_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    result_folder = os.path.join(figure3_path, "result", "visibility_top1")
    figure_folder = os.path.join(figure3_path, "figure")

    paper_file = os.path.join(
        result_folder, "figure3_top1_count50_paper_metrics.csv.gz"
    )
    summary_file = os.path.join(
        result_folder, "figure3_top1_count_distribution_summary.csv"
    )
    bin_file = os.path.join(
        result_folder, "figure3_top1_count_distribution_bins.csv"
    )
    pdf_path = os.path.join(
        figure_folder, "Figure3_top1_count_distribution.pdf"
    )
    png_path = os.path.join(
        figure_folder, "Figure3_top1_count_distribution.png"
    )

    start_year = 1970
    end_year = 2020
    bins = 20
    log_y = True
    
    period_order = [
        "Print-dominant era (1970-1979)",
        "Digitization transition (1980-2003)",
        "Google Scholar era (2004-2020)",
    ]
    colors = {
        "Print-dominant era (1970-1979)": "#3B6FB6",
        "Digitization transition (1980-2003)": "#D17A22",
        "Google Scholar era (2004-2020)": "#2A8C82",
    }

    os.makedirs(figure_folder, exist_ok=True)
    papers = load_paper_counts(paper_file, start_year, end_year)
    distributions, bin_edges, bin_centers, summary, bin_data = (
        calculate_distributions(
            papers,
            period_order,
            bins,
        )
    )
    summary.to_csv(summary_file, index=False)
    bin_data.to_csv(bin_file, index=False)
    plot_distribution(
        distributions,
        bin_edges,
        bin_centers,
        summary,
        period_order,
        colors,
        log_y,
        pdf_path,
        png_path,
    )
    print("Top-1% reference-count distribution figure was saved successfully.")
