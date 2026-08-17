#!/usr/bin/env python3
"""Plot yearly C10 (10-year window) citation Gini coefficients for MAG and
OpenAlex.

Visual style is identical to figure1/code/plot_c5_gini.py; only the citation
window differs (C10 vs C5). Because a full 10-year window is required, cohorts
are truncated at (last cohort year - 10), analogous to the -5 cut used for C5.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

WINDOW_CUT = 10  # drop cohorts without a complete 10-year citation window


def load_gini_data(input_file, groups):
    """Load the saved Gini table and retain the requested percentile groups."""
    data = pd.read_csv(input_file)
    data = data[data["Group"].isin(groups)].copy()
    data = data.sort_values(["Group", "Year"])
    return data


def style_axis(ax, show_xlabel=True, show_ylabel=True):
    """Apply a clean journal-style appearance to one axis."""
    ax.set_xlim(1950, 2020)
    ax.set_xticks([1960, 1980, 2000, 2020])
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)

    if show_xlabel:
        ax.set_xlabel("Publication year", fontsize=9)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("Gini coefficient", fontsize=9)


def add_google_scholar_marker(ax, text_height=0.99):
    """Mark the launch year of Google Scholar with a neutral shaded band."""
    ax.axvspan(2003.5, 2004.5, color="#8C8C8C", alpha=0.38, linewidth=0)
    ax.annotate(
        "Google Scholar\nlaunched (2004)",
        xy=(2004, text_height),
        xytext=(1991, text_height),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        ha="center",
        va="center",
        fontsize=6.5,
        color="#4D4D4D",
        arrowprops={"arrowstyle": "-", "color": "#6E6E6E", "lw": 0.8},
    )


def add_digitization_marker(ax, text_height=0.94):
    """Mark the start of the digitization and networked-publishing transition."""
    ax.axvspan(1979.5, 1980.5, color="#AAB2BD", alpha=0.38, linewidth=0)
    ax.annotate(
        "Digitization\ntransition (1980)",
        xy=(1980, text_height),
        xytext=(1963, text_height),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        ha="center",
        va="center",
        fontsize=6.5,
        color="#4D4D4D",
        arrowprops={"arrowstyle": "-", "color": "#7B8490", "lw": 0.8},
    )


def add_discovery_eras(ax):
    """Shade the publishing-transition and Google-Scholar periods."""
    ax.axvspan(1980, 2004, color="#D9DDE3", alpha=0.60, linewidth=0)
    ax.axvspan(2004, 2020, color="#EFEFEF", alpha=0.78, linewidth=0)


def plot_group(
    ax, data, group, color, show_xlabel=True, show_ylabel=True, end_year=None
):
    """Plot one percentile group's yearly Gini trajectory."""
    group_data = data[data["Group"] == group]
    if end_year is not None:
        group_data = group_data[group_data["Year"] <= end_year]
    ax.plot(
        group_data["Year"],
        group_data["Gini_C10"],
        color=color,
        linewidth=1.8,
        solid_capstyle="round",
    )
    display_titles = {
        "Top 0-1%": "Top 1%",
        "Top 1-5%": "Top 1–5%",
        "Top 5-10%": "Top 5–10%",
        "Top 10-15%": "Top 10–15%",
    }
    ax.set_title(display_titles[group], fontsize=9, pad=6)
    style_axis(ax, show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def plot_dataset_row(fig, outer_spec, data, dataset_name, colors, top_row):
    """Plot the large top-1% panel and three vertically stacked panels."""
    left_ax = fig.add_subplot(outer_spec[0])
    top_one_last_year = data.loc[data["Group"] == "Top 0-1%", "Year"].max()
    end_year = top_one_last_year - WINDOW_CUT
    plot_group(
        left_ax,
        data,
        "Top 0-1%",
        colors["Top 0-1%"],
        show_xlabel=True,
        show_ylabel=True,
        end_year=end_year,
    )
    add_digitization_marker(left_ax)
    add_google_scholar_marker(left_ax, text_height=0.95)

    right_grid = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_spec[1], hspace=0.62
    )
    small_axes = []
    small_groups = ["Top 1-5%", "Top 5-10%", "Top 10-15%"]

    for index, group in enumerate(small_groups):
        ax = fig.add_subplot(right_grid[index, 0])
        plot_group(
            ax,
            data,
            group,
            colors[group],
            show_xlabel=index == len(small_groups) - 1,
            show_ylabel=index == 1,
            end_year=end_year,
        )
        add_discovery_eras(ax)
        small_axes.append(ax)

    return left_ax, small_axes


def add_panel_labels(fig, axes, labels):
    """Add panel letters above the four main panel groups."""
    for ax, label in zip(axes, labels):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.052, position.y1 + 0.010, label,
            fontsize=11, fontweight="bold", ha="left", va="bottom"
        )


def add_era_legend(fig):
    """Add a compact two-line legend for the three historical periods."""
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#FFFFFF", edgecolor="#BDBDBD",
            linewidth=0.6
        ),
        plt.Rectangle((0, 0), 1, 1, color="#D9DDE3", alpha=0.90, linewidth=0),
        plt.Rectangle((0, 0), 1, 1, color="#EFEFEF", alpha=0.95, linewidth=0),
    ]
    labels = [
        "Print-dominant era (1950–1979)",
        "Digitization and networked-publishing transition (1980–2003)",
        "Google Scholar era (2004 onward)",
    ]
    fig.legend(
        handles[:2], labels[:2], loc="lower center",
        bbox_to_anchor=(0.50, 0.040), ncol=2, frameon=False, fontsize=7.0,
        handlelength=1.3, columnspacing=1.0
    )
    fig.legend(
        handles[2:], labels[2:], loc="lower center",
        bbox_to_anchor=(0.50, 0.014), ncol=1, frameon=False, fontsize=7.0,
        handlelength=1.3
    )


def make_figure(mag_data, openalex_data, output_path, colors):
    """Create and save the complete 2-by-2 unequal-panel figure."""
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelcolor": "#262626",
        "xtick.color": "#262626",
        "ytick.color": "#262626",
        "text.color": "#262626",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(6, 6))
    outer = GridSpec(
        2, 2, figure=fig, width_ratios=[1.2, 1.0],
        height_ratios=[1, 1], wspace=0.34, hspace=0.30
    )

    mag_left, mag_small = plot_dataset_row(
        fig, (outer[0, 0], outer[0, 1]), mag_data, "MAG", colors, top_row=True
    )
    oa_left, oa_small = plot_dataset_row(
        fig, (outer[1, 0], outer[1, 1]), openalex_data, "OpenAlex", colors,
        top_row=False
    )

    fig.subplots_adjust(left=0.110, right=0.965, top=0.920, bottom=0.175)

    add_panel_labels(
        fig,
        [mag_left, mag_small[0], oa_left, oa_small[0]],
        ["(a)  MAG", "(b)", "(c)  OpenAlex", "(d)"],
    )
    add_era_legend(fig)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, "FigureS1_gini_C10.png"), dpi=600)7
    fig.savefig(os.path.join(output_path, "FigureS1_gini_C10.pdf"))


if __name__ == "__main__":

    mag_file = "/Volumes/lydisk/work/work11/google_scholar/figureS1/result/gini/c10_gini_mag.csv"
    openalex_file = "/Volumes/lydisk/work/work11/google_scholar/figureS1/result/gini/c10_gini_openalex.csv"
    output_path = "/Volumes/lydisk/work/work11/google_scholar/figureS1/figure"
    
    groups = ["Top 0-1%", "Top 1-5%", "Top 5-10%", "Top 10-15%"]
    colors = {
        "Top 0-1%": "#3C5488",
        "Top 1-5%": "#E64B35",
        "Top 5-10%": "#00A087",
        "Top 10-15%": "#C49A32",
    }
    
    mag_data = load_gini_data(mag_file, groups)
    openalex_data = load_gini_data(openalex_file, groups)
    make_figure(mag_data, openalex_data, output_path, colors)
