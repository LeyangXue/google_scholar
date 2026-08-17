#!/usr/bin/env python3
"""Calculate yearly C10 (10-year window) citation Gini coefficients by
citation-percentile group.

This is the 10-year-window robustness counterpart of figure1/code/
calculate_c5_gini.py. Everything is identical except the citation metric
(C10 instead of C5), so any difference in the resulting figure reflects the
window length alone.
"""

import os

import numpy as np
import pandas as pd


def gini_coefficient(values):
    """Calculate the Gini coefficient with an efficient sorted-array formula."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan
    if np.any(values < 0):
        raise ValueError("Citation counts must be non-negative.")

    total = values.sum()
    if total == 0:
        return 0.0

    values.sort()
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return (2.0 * np.dot(ranks, values) / (values.size * total)
            - (values.size + 1.0) / values.size)


def select_percentile_group(citations, lower, upper):
    """Select a rank-based interval from descending-sorted citations."""
    start = int(np.floor(citations.size * lower))
    stop = int(np.floor(citations.size * upper))
    return citations[start:stop]


def calculate_dataset_gini(data_path, dataset_name, years, percentile_groups):
    """Load annual C10 data and calculate Gini coefficients for all groups."""
    rows = []

    for year in years:
        input_file = os.path.join(data_path, f"citation_year{year}.csv")
        print(f"[{dataset_name}] Processing {year}: {input_file}")

        if not os.path.exists(input_file):
            print(f"[{dataset_name}] Missing file for {year}; skipped.")
            continue

        citation_data = pd.read_csv(input_file, usecols=["C10"])
        citations = pd.to_numeric(citation_data["C10"], errors="coerce").dropna().to_numpy()
        citations = np.sort(citations.astype(np.float64))[::-1]

        for group_name, bounds in percentile_groups.items():
            lower, upper = bounds
            group_values = select_percentile_group(citations, lower, upper)

            rows.append({
                "Dataset": dataset_name,
                "Year": year,
                "Group": group_name,
                "Lower_percentile": lower * 100,
                "Upper_percentile": upper * 100,
                "N_papers": group_values.size,
                "Gini_C10": gini_coefficient(group_values),
            })

    return pd.DataFrame(rows)


def save_results(results, output_path, dataset_name):
    """Save one dataset's yearly Gini results as a CSV file."""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"c10_gini_{dataset_name.lower()}.csv")
    results.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


if __name__ == "__main__":

    mag_data_path = "/Volumes/lydisk/work/work11/google_scholar/figure1/result/mag/year_citation_journal"
    openalex_data_path = "/Volumes/lydisk/work/work11/google_scholar/figure1/result/openalex/year_citation_journal"
    output_path = "/Volumes/lydisk/work/work11/google_scholar/figureS1/result/gini"

    start_year = 1950
    end_year = 2020
    years = range(start_year, end_year + 1)

    percentile_groups = {
        "Top 0-1%": (0.00, 0.01),
        "Top 1-5%": (0.01, 0.05),
        "Top 5-10%": (0.05, 0.10),
        "Top 10-15%": (0.10, 0.15),
    }

    mag_results = calculate_dataset_gini(
        mag_data_path, "MAG", years, percentile_groups
    )
    save_results(mag_results, output_path, "MAG")

    openalex_results = calculate_dataset_gini(
        openalex_data_path, "OpenAlex", years, percentile_groups
    )
    save_results(openalex_results, output_path, "OpenAlex")

    combined_results = pd.concat(
        [mag_results, openalex_results], ignore_index=True
    )
    combined_file = os.path.join(output_path, "c10_gini_combined.csv")
    combined_results.to_csv(combined_file, index=False)
    print(f"Saved: {combined_file}")
