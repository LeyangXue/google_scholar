#!/usr/bin/env python3
"""Plot Figure 2: elite retention and survival for MAG and OpenAlex."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def period_name(year):
    """Return the historical period used consistently across the figures."""
    if year <= 1979:
        return "Print-dominant era (1950–1979)"
    if year <= 2003:
        return "Digitization and networked-publishing transition (1980–2003)"
    return "Google Scholar era (2004 onward)"


def moving_block_bootstrap(values, iterations, block_length, rng):
    """Return moving-block bootstrap estimates of the annual-series mean."""
    values = np.asarray(values, dtype=float)
    n_values = values.size
    if n_values == 0:
        return np.array([])
    if n_values == 1:
        return np.repeat(values[0], iterations)

    block_length = min(block_length, n_values)
    n_blocks = int(math.ceil(n_values / block_length))
    max_start = n_values - block_length
    estimates = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([
            values[start:start + block_length] for start in starts
        ])[:n_values]
        estimates[iteration] = sample.mean()
    return estimates


def calculate_period_statistics(
    retention,
    period_order,
    bootstrap_iterations,
    bootstrap_block_length,
    random_seed,
):
    """Recalculate period means and confidence intervals from yearly retention."""
    data = retention.copy()
    data["Period"] = data["Publication_year"].map(period_name)
    data["Period"] = pd.Categorical(
        data["Period"], categories=period_order, ordered=True
    )
    rng = np.random.default_rng(random_seed)
    rows = []

    for (dataset, measure, period), group in data.groupby(
        ["Dataset", "Measure", "Period"], observed=True, sort=False
    ):
        group = group.sort_values("Publication_year")
        years = group["Publication_year"].to_numpy(dtype=int)
        values = group["Retention"].to_numpy(dtype=float)
        bootstrap = moving_block_bootstrap(
            values,
            bootstrap_iterations,
            bootstrap_block_length,
            rng,
        )
        rows.append({
            "Dataset": dataset,
            "Measure": measure,
            "Period": str(period),
            "Start_year": int(years.min()),
            "End_year": int(years.max()),
            "N_years": len(years),
            "Mean_retention": values.mean(),
            "CI_lower": np.quantile(bootstrap, 0.025),
            "CI_upper": np.quantile(bootstrap, 0.975),
        })
    return data, pd.DataFrame(rows)


def calculate_kaplan_meier(paper_level_files, period_order):
    """Recalculate period-specific Kaplan-Meier curves from paper-level data."""
    rows = []

    for paper_level_file in paper_level_files:
        data = pd.read_csv(
            paper_level_file,
            usecols=[
                "Dataset",
                "Publication_year",
                "Followup_time",
                "Event",
            ],
        )
        data["Period"] = data["Publication_year"].map(period_name)
        data["Period"] = pd.Categorical(
            data["Period"], categories=period_order, ordered=True
        )

        for (dataset, period), group in data.groupby(
            ["Dataset", "Period"], observed=True, sort=False
        ):
            duration = group["Followup_time"].to_numpy(dtype=int)
            event = group["Event"].to_numpy(dtype=int)
            survival = 1.0
            greenwood_sum = 0.0
            max_time = int(duration.max())

            for time in range(max_time + 1):
                at_risk = int(np.count_nonzero(duration >= time))
                events = int(np.count_nonzero((duration == time) & (event == 1)))
                censored = int(np.count_nonzero((duration == time) & (event == 0)))

                if time > 0 and at_risk > 0 and events > 0:
                    survival *= 1.0 - events / at_risk
                    if at_risk > events:
                        greenwood_sum += events / (at_risk * (at_risk - events))

                standard_error = survival * math.sqrt(greenwood_sum)
                rows.append({
                    "Dataset": dataset,
                    "Period": str(period),
                    "Time_since_age_2": time,
                    "At_risk": at_risk,
                    "Events": events,
                    "Censored": censored,
                    "Survival": survival,
                    "CI_lower": max(0.0, survival - 1.96 * standard_error),
                    "CI_upper": min(1.0, survival + 1.96 * standard_error),
                })
        del data
    return pd.DataFrame(rows)


def load_results(
    retention_file,
    paper_level_files,
    period_order,
    bootstrap_iterations,
    bootstrap_block_length,
    random_seed,
):
    """Load yearly retention and recalculate all period-specific statistics."""
    retention = pd.read_csv(retention_file)

    # MAG citation histories end in 2019. A cohort therefore requires a
    # publication year of 2009 or earlier to provide complete follow-up from
    # citation age 5 through citation age 10.
    incomplete_mag_ten_year_followup = (
        (retention["Dataset"] == "MAG")
        & (retention["Measure"] == "Five-year forward retention")
        & (retention["Publication_year"] > 2009)
    )
    retention = retention.loc[~incomplete_mag_ten_year_followup].copy()

    retention, statistics = calculate_period_statistics(
        retention,
        period_order,
        bootstrap_iterations,
        bootstrap_block_length,
        random_seed,
    )
    survival = calculate_kaplan_meier(paper_level_files, period_order)
    return retention, statistics, survival


def style_axis(ax):
    """Apply a clean journal-style axis appearance."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)


def add_historical_periods(ax, end_year):
    """Add restrained background shading for the two digital-search periods."""
    ax.axvspan(1980, 2004, color="#D9DDE3", alpha=0.60, linewidth=0, zorder=0)
    ax.axvspan(2004, end_year, color="#EFEFEF", alpha=0.78, linewidth=0, zorder=0)


def plot_retention_panel(
    ax,
    retention,
    statistics,
    dataset,
    measure,
    line_color,
    show_ylabel,
    y_max,
):
    """Plot yearly retention with period means and bootstrap confidence intervals."""
    annual = retention[
        (retention["Dataset"] == dataset)
        & (retention["Measure"] == measure)
    ].sort_values("Publication_year")
    period_stats = statistics[
        (statistics["Dataset"] == dataset)
        & (statistics["Measure"] == measure)
    ]

    add_historical_periods(ax, 2020)
    ax.plot(
        annual["Publication_year"],
        annual["Retention"],
        color=line_color,
        linewidth=1.8,
        solid_capstyle="round",
        zorder=3,
    )

    for _, row in period_stats.iterrows():
        start = row["Start_year"]
        end = row["End_year"]
        mean = row["Mean_retention"]
        lower = row["CI_lower"]
        upper = row["CI_upper"]
        ax.fill_between(
            [start, end],
            [lower, lower],
            [upper, upper],
            color=line_color,
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )
        ax.hlines(
            mean,
            start,
            end,
            color=line_color,
            linewidth=3.0,
            alpha=0.76,
            zorder=2,
        )
        label_offset = 0.018
        if (
            dataset == "MAG"
            and measure == "Five-year forward retention"
            and row["Period"] == "Google Scholar era (2004 onward)"
        ):
            label_offset = 0.038
        ax.text(
            (start + end) / 2,
            upper + label_offset,
            f"{mean:.2f}",
            color="#4D4D4D",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_xlim(1950, 2020)
    ax.set_ylim(0.56, y_max)
    ax.set_xticks([1960, 1980, 2000, 2020])
    ax.set_xlabel("Publication year", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Top-1% retention probability", fontsize=9)

    style_axis(ax)


def plot_survival_panel(
    ax,
    survival,
    dataset,
    period_order,
    period_colors,
    period_labels,
    max_survival_time,
    show_ylabel,
    show_legend,
):
    """Plot period-specific Kaplan-Meier top-1% survival curves."""
    dataset_data = survival[
        (survival["Dataset"] == dataset)
        & (survival["Time_since_age_2"] <= max_survival_time)
    ]

    for period in period_order:
        curve = dataset_data[dataset_data["Period"] == period].sort_values(
            "Time_since_age_2"
        )
        color = period_colors[period]
        ax.fill_between(
            curve["Time_since_age_2"],
            curve["CI_lower"],
            curve["CI_upper"],
            step="post",
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax.step(
            curve["Time_since_age_2"],
            curve["Survival"],
            where="post",
            color=color,
            linewidth=2.0,
            label=period_labels[period],
        )

    ax.set_xlim(0, max_survival_time)
    ax.set_ylim(0.35, 1.01)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Years since top-1% status at citation age 2", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Top-1% survival probability", fontsize=9)
    style_axis(ax)

    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.02, 1.02),
            frameon=False,
            fontsize=6.0,
            handlelength=1.8,
            handletextpad=0.5,
            labelspacing=0.35,
            borderaxespad=0.0,
        )


def add_panel_labels(fig, axes, labels):
    """Place panel labels at consistent figure-level heights."""
    for ax, label in zip(axes, labels):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.045,
            position.y1 + 0.012,
            label,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def add_period_legend(fig):
    """Add a figure-level legend for calendar-period background shading."""
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
    fig.legend(
        handles[:3],
        labels[:3],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.026),
        ncol=3,
        frameon=False,
        fontsize=7.1,
        handlelength=1.5,
        columnspacing=1.0,
    )
    fig.legend(
        handles[3:],
        labels[3:],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.002),
        ncol=1,
        frameon=False,
        fontsize=7.1,
        handlelength=1.8,
    )


def make_figure(
    retention,
    statistics,
    survival,
    output_path,
    dataset_colors,
    period_order,
    period_colors,
    period_labels,
    max_survival_time,
):
    """Create and save the complete 2-by-3 Figure 2."""
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelcolor": "#262626",
        "xtick.color": "#262626",
        "ytick.color": "#262626",
        "text.color": "#262626",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
    measures = [
        "Two-year forward retention",
        "Five-year forward retention",
    ]
    datasets = ["MAG", "OpenAlex"]

    for row, dataset in enumerate(datasets):
        for column, measure in enumerate(measures):
            plot_retention_panel(
                axes[row, column],
                retention,
                statistics,
                dataset,
                measure,
                dataset_colors[dataset],
                show_ylabel=column == 0,
                y_max=0.90,
            )
        plot_survival_panel(
            axes[row, 2],
            survival,
            dataset,
            period_order,
            period_colors,
            period_labels,
            max_survival_time,
            show_ylabel=True,
            show_legend=row == 0,
        )

    column_titles = [
        "Top-1% retention (age 2→4)",
        "Top-1% retention (age 5→10)",
        "Top-1% survival (from age 2)",
    ]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=9.6, pad=10)

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.91,
        bottom=0.15,
        wspace=0.28,
        hspace=0.31,
    )
    add_panel_labels(
        fig,
        axes.flatten(),
        ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"],
    )
    for row, dataset in enumerate(datasets):
        pos = axes[row, 0].get_position()
        fig.text(
            0.026,
            (pos.y0 + pos.y1) / 2,
            dataset,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    add_period_legend(fig)

    os.makedirs(output_path, exist_ok=True)
    fig.savefig(
        os.path.join(output_path, "Figure2_elite_persistence.png"),
        dpi=600,
    )
    fig.savefig(os.path.join(output_path, "Figure2_elite_persistence.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    
    result_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/result/retention_survival"
    retention_file = os.path.join(
        result_path, "elite_retention_yearly_combined.csv"
    )
    paper_level_files = [
        os.path.join(result_path, "elite_survival_paper_level_mag.csv"),
        os.path.join(result_path, "elite_survival_paper_level_openalex.csv"),
    ]
    output_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/figure"

    dataset_colors = {
        "MAG": "#3C5488",
        "OpenAlex": "#E64B35",
    }
    period_order = [
        "Print-dominant era (1950–1979)",
        "Digitization and networked-publishing transition (1980–2003)",
        "Google Scholar era (2004 onward)",
    ]
    period_colors = {
        "Print-dominant era (1950–1979)": "#777777",
        "Digitization and networked-publishing transition (1980–2003)": "#00A087",
        "Google Scholar era (2004 onward)": "#E64B35",
    }
    period_labels = {
        "Print-dominant era (1950–1979)": "Print-dominant (1950–1979)",
        "Digitization and networked-publishing transition (1980–2003)":
            "Digitization transition (1980–2003)",
        "Google Scholar era (2004 onward)": "Google Scholar era (2004 onward)",
    }
    max_survival_time = 15
    bootstrap_iterations = 2000
    bootstrap_block_length = 5
    random_seed = 20260712

    retention, statistics, survival = load_results(
        retention_file,
        paper_level_files,
        period_order,
        bootstrap_iterations,
        bootstrap_block_length,
        random_seed,
    )
    make_figure(
        retention,
        statistics,
        survival,
        output_path,
        dataset_colors,
        period_order,
        period_colors,
        period_labels,
        max_survival_time,
    )
