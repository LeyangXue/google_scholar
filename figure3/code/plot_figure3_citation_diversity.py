import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_year_sample(path):
    """Extract citing year and sample number from a file name."""
    match = re.search(r"topic_distance_(\d{4})_k(\d+)\.csv$", path)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def to_distance(values, method):
    """Orient every metric so that larger values mean greater semantic distance."""
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if method in {"embedding", "jaccard"}:
        values = 1.0 - values
    return values


def find_predigital_threshold(folder, method, quantile=0.90):
    """Calculate one fixed pre-digital distance threshold for all later years."""
    distance_parts = []

    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        year, sample = parse_year_sample(path)
        if year is None or year >= 1990:
            continue

        frame = pd.read_csv(path, usecols=["TopicDistance"])
        distance = to_distance(frame["TopicDistance"], method)
        distance = distance[np.isfinite(distance)]
        distance_parts.append(distance)

    if not distance_parts:
        raise ValueError("No valid pre-digital observations were found.")

    distances = np.concatenate(distance_parts)
    threshold = float(np.quantile(distances, quantile))
    return threshold, len(distances)


def calculate_sample_metrics(path, method, distant_threshold=None, min_references=5):
    """Calculate paper-level outcomes first and then average them within a sample."""
    year, sample = parse_year_sample(path)
    frame = pd.read_csv(path, usecols=["CitingPublicationId", "TopicDistance"])
    frame["distance"] = to_distance(frame["TopicDistance"], method)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["CitingPublicationId", "distance"]
    )

    if frame.empty:
        return None

    frame["distance_sq"] = frame["distance"] ** 2
    if distant_threshold is not None:
        frame["is_distant"] = (frame["distance"] > distant_threshold).astype(float)

    columns = {
        "reference_count": ("distance", "size"),
        "distance_sum": ("distance", "sum"),
        "distance_sq_sum": ("distance_sq", "sum"),
    }
    if distant_threshold is not None:
        columns["distant_count"] = ("is_distant", "sum")

    papers = frame.groupby("CitingPublicationId", sort=False).agg(**columns)
    papers["mean_distance"] = papers["distance_sum"] / papers["reference_count"]
    papers["distance_variance"] = (
        papers["distance_sq_sum"] / papers["reference_count"]
        - papers["mean_distance"] ** 2
    ).clip(lower=0.0)

    if distant_threshold is not None:
        papers["distant_share"] = papers["distant_count"] / papers["reference_count"]
    else:
        papers["distant_share"] = np.nan

    eligible = papers["reference_count"] >= min_references
    return {
        "method": method,
        "year": year,
        "sample": sample,
        "mean_distance": papers["mean_distance"].mean(),
        "distant_share": papers["distant_share"].mean(),
        "heterogeneity": papers.loc[eligible, "distance_variance"].mean(),
        "n_papers": len(papers),
        "n_heterogeneity_papers": int(eligible.sum()),
        "n_references": int(papers["reference_count"].sum()),
    }


def summarize_all_samples(method_folders, threshold, threshold_method, min_references=5):
    """Process all methods while preserving year and sample identities."""
    records = []
    for method, folder in method_folders.items():
        paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
        for number, path in enumerate(paths, start=1):
            metric_threshold = threshold if method == threshold_method else None
            record = calculate_sample_metrics(
                path,
                method,
                distant_threshold=metric_threshold,
                min_references=min_references,
            )
            if record is not None:
                records.append(record)
            if number % 100 == 0 or number == len(paths):
                print(f"Processed {method}: {number}/{len(paths)} files")
    return pd.DataFrame(records)


def t_multiplier(n):
    """Return a 95% two-sided t multiplier, with a normal fallback."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
    except ImportError:
        return 1.96 if n > 1 else np.nan


def summarize_by_year(samples):
    """Average the 30 sample estimates within each year and calculate 95% CIs."""
    rows = []
    metrics = ["mean_distance", "distant_share", "heterogeneity"]
    for (method, year), group in samples.groupby(["method", "year"]):
        row = {"method": method, "year": int(year)}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            se = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan
            half_width = t_multiplier(n) * se if n > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_low"] = mean - half_width
            row[f"{metric}_high"] = mean + half_width
            row[f"{metric}_n_samples"] = n
        row["n_papers"] = group["n_papers"].sum()
        row["n_references"] = group["n_references"].sum()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "year"])


def period_name(year):
    """Assign a year to one of the three scholarly-search periods."""
    if year < 1990:
        return "Pre-digital (1950–1989)"
    if year < 2004:
        return "Digital publishing (1990–2003)"
    return "Google Scholar era (2004 onward)"


def summarize_by_period(years):
    """Give each observed year equal weight when calculating period means."""
    frame = years.copy()
    frame["period"] = frame["year"].map(period_name)
    rows = []
    metrics = ["mean_distance_mean", "distant_share_mean", "heterogeneity_mean"]
    for (method, period), group in frame.groupby(["method", "period"], sort=False):
        row = {"method": method, "period": period, "n_years": len(group)}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[metric] = np.mean(values) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


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
        }
    )


def shade_periods(ax):
    """Mark the digital-publishing and Google Scholar periods."""
    ax.axvspan(1990, 2004, color="#8F969C", alpha=0.18, linewidth=0, zorder=0)
    ax.axvspan(2004, 2022.8, color="#D7D9DB", alpha=0.34, linewidth=0, zorder=0)
    ax.axvline(2004, color="#71777C", linewidth=0.8, zorder=1)


def draw_time_panel(ax, years, metric, ylabel, title, color):
    """Draw a yearly estimate, sample-based CI, and three period means."""
    shade_periods(ax)
    x = years["year"].to_numpy(dtype=float)
    y = years[f"{metric}_mean"].to_numpy(dtype=float)
    low = years[f"{metric}_low"].to_numpy(dtype=float)
    high = years[f"{metric}_high"].to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1.8)

    for start, end in [(1950, 1989), (1990, 2003), (2004, 2022)]:
        selected = years[(years["year"] >= start) & (years["year"] <= end)]
        mean = selected[f"{metric}_mean"].mean()
        if np.isfinite(mean):
            ax.hlines(mean, start, end, color=color, linewidth=1.0, linestyle=(0, (3, 2)))

    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1950, 2022)
    ax.set_xticks([1950, 1970, 1990, 2004, 2022])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def plot_figure(years, figure_path, png_path):
    """Create the horizontal 1 by 3 Figure 3 layout."""
    set_plot_style()
    primary = years[years["method"] == "masscenter"].sort_values("year")
    main_color = "#3B6FB6"

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.65), constrained_layout=False)
    draw_time_panel(
        axes[0], primary, "mean_distance", "Mean mass-center distance",
        "(a) Mean semantic distance", main_color,
    )
    draw_time_panel(
        axes[1], primary, "distant_share", "Distant-reference share",
        "(b) Distant-reference share", main_color,
    )
    draw_time_panel(
        axes[2], primary, "heterogeneity", "Mean within-paper variance",
        "(c) Within-bibliography heterogeneity", main_color,
    )

    for ax in axes:
        ax.set_xlabel("Year")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    period_handles = [
        Line2D([0], [0], color="#8F969C", linewidth=6, alpha=0.25,
               label="Digital publishing (1990–2003)"),
        Line2D([0], [0], color="#D7D9DB", linewidth=6, alpha=0.65,
               label="Google Scholar era (2004 onward)"),
    ]
    figure.legend(
        handles=period_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.25, wspace=0.32)
    figure.savefig(figure_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    base_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    data_path = os.path.join(base_path, "result", "openalex")
    summary_path = os.path.join(base_path, "result", "summary")
    figure_path = os.path.join(base_path, "figure")

    masscenter_folder = os.path.join(data_path, "year_topic_distance_masscenter")
    method_folders = {
        "masscenter": masscenter_folder,
    }

    min_references = 5
    threshold_quantile = 0.90
    threshold_method = "masscenter"
    recalculate = False

    os.makedirs(summary_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    sample_file = os.path.join(summary_path, "figure3_sample_metrics.csv")
    year_file = os.path.join(summary_path, "figure3_year_metrics.csv")
    period_file = os.path.join(summary_path, "figure3_period_metrics.csv")
    threshold_file = os.path.join(summary_path, "figure3_threshold.csv")

    if recalculate or not all(os.path.exists(path) for path in [sample_file, year_file, period_file, threshold_file]):
        threshold, threshold_n = find_predigital_threshold(
            masscenter_folder,
            method=threshold_method,
            quantile=threshold_quantile,
        )
        print(f"Fixed pre-digital 90th-percentile threshold: {threshold:.6f}")

        samples = summarize_all_samples(
            method_folders,
            threshold,
            threshold_method=threshold_method,
            min_references=min_references,
        )
        years = summarize_by_year(samples)
        periods = summarize_by_period(years)

        samples.to_csv(sample_file, index=False)
        years.to_csv(year_file, index=False)
        periods.to_csv(period_file, index=False)
        pd.DataFrame(
            [{
                "metric": "masscenter_distance",
                "reference_period": "1950–1989",
                "quantile": threshold_quantile,
                "threshold": threshold,
                "n_reference_edges": threshold_n,
            }]
        ).to_csv(threshold_file, index=False)
    else:
        years = pd.read_csv(year_file)

    plot_figure(
        years,
        os.path.join(figure_path, "Figure3_citation_diversity.pdf"),
        os.path.join(figure_path, "Figure3_citation_diversity.png"),
    )
    print("Figure 3 summaries and plots were saved successfully.")
