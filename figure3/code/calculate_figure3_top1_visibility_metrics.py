import os

import numpy as np
import pandas as pd

import calculate_figure3_visibility_metrics as base


def period_name(year):
    """Assign a citing year to the updated historical periods."""
    if year <= 1979:
        return "Print-dominant era (1950-1979)"
    if year <= 2003:
        return "Digitization and networked-publishing transition (1980-2003)"
    return "Google Scholar era (2004 onward)"


def summarize_periods_across_samples(samples):
    """Calculate period means and 95% CIs across 30 sample-level estimates."""
    sample_periods = (
        samples.groupby(["period", "sample"], as_index=False)[
            "high_visibility_gap"
        ]
        .mean()
    )
    rows = []
    for period, group in sample_periods.groupby("period", sort=False):
        values = group["high_visibility_gap"].dropna().to_numpy(dtype=float)
        n = len(values)
        mean = np.mean(values) if n else np.nan
        standard_error = (
            np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        )
        half_width = (
            base.t_multiplier(n) * standard_error if n > 1 else np.nan
        )
        rows.append(
            {
                "period": period,
                "n_samples": n,
                "high_visibility_gap_mean": mean,
                "high_visibility_gap_low": mean - half_width,
                "high_visibility_gap_high": mean + half_width,
                "uncertainty_definition": (
                    "95% t CI across sample-level period means"
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_top1_visibility_results(
    figure3_path,
    cohort_folder,
    citing_years,
    min_visibility_age,
    max_visibility_age,
    high_visibility_cutoff,
    closest_fraction,
    min_matched_references,
    save_paper_level,
):
    """Calculate Figure 3 metrics using the top 1% visibility definition."""
    distance_folder = os.path.join(
        figure3_path,
        "result",
        "openalex",
        "year_topic_distance_embeding",
    )
    output_folder = os.path.join(figure3_path, "result", "visibility_top1")
    os.makedirs(output_folder, exist_ok=True)

    paper_parts = []
    sample_rows = []
    cohort_quality_rows = []

    for citing_year in citing_years:
        print(f"Processing citing year {citing_year}", flush=True)
        edges = base.load_year_edges(distance_folder, citing_year)
        visibility_lookup, cohort_quality = base.build_visibility_lookup(
            edges,
            citing_year,
            cohort_folder,
            min_visibility_age,
            max_visibility_age,
            high_visibility_cutoff,
        )
        papers, matched, eligible = base.calculate_paper_metrics(
            edges,
            citing_year,
            visibility_lookup,
            min_visibility_age,
            max_visibility_age,
            closest_fraction,
            min_matched_references,
        )
        year_sample_rows = base.calculate_sample_metrics(
            papers,
            edges,
            matched,
            eligible,
            citing_year,
        )
        for row in year_sample_rows:
            row["period"] = period_name(citing_year)
        sample_rows.extend(year_sample_rows)
        cohort_quality_rows.extend(cohort_quality)
        if save_paper_level:
            paper_parts.append(papers)

    samples = pd.DataFrame(sample_rows).sort_values(["year", "sample"])
    years = base.summarize_years(samples)
    years["period"] = years["year"].map(period_name)
    periods = base.summarize_periods(years)
    period_sample_ci = summarize_periods_across_samples(samples)
    cohort_quality = pd.DataFrame(cohort_quality_rows).sort_values(
        ["citing_year", "citation_age_at_citation"]
    )

    samples.to_csv(
        os.path.join(output_folder, "figure3_top1_sample_metrics.csv"),
        index=False,
    )
    years.to_csv(
        os.path.join(output_folder, "figure3_top1_year_metrics.csv"),
        index=False,
    )
    periods.to_csv(
        os.path.join(output_folder, "figure3_top1_period_metrics.csv"),
        index=False,
    )
    period_sample_ci.to_csv(
        os.path.join(output_folder, "figure3_top1_period_sample_ci.csv"),
        index=False,
    )
    cohort_quality.to_csv(
        os.path.join(output_folder, "figure3_top1_cohort_quality.csv"),
        index=False,
    )

    if save_paper_level:
        papers = pd.concat(paper_parts, ignore_index=True)
        papers.to_csv(
            os.path.join(output_folder, "figure3_top1_paper_metrics.csv.gz"),
            index=False,
            compression="gzip",
        )

    parameters = pd.DataFrame(
        [
            {
                "semantic_metric": "cosine_similarity",
                "visibility_source": "Figure 2 annual C0 onward",
                "visibility_reference_group": "same publication year and age",
                "prior_citation_definition": "sum C0 through C(age-1)",
                "min_citation_age_at_citation": min_visibility_age,
                "max_citation_age_at_citation": max_visibility_age,
                "high_visibility_percentile_cutoff": high_visibility_cutoff,
                "high_visibility_definition": "age-standardized percentile above 0.99",
                "closest_reference_fraction": closest_fraction,
                "min_matched_references_for_closest_analysis": min_matched_references,
                "paper_weighting": "equal",
                "sample_weighting_within_year": "equal",
                "year_weighting_within_period": "equal",
                "period_gap_uncertainty": (
                    "95% t CI across 30 sample-level period means"
                ),
                "period_1": "Print-dominant era (1950-1979)",
                "period_2": "Digitization and networked-publishing transition (1980-2003)",
                "period_3": "Google Scholar era (2004 onward)",
            }
        ]
    )
    parameters.to_csv(
        os.path.join(output_folder, "figure3_top1_parameters.csv"),
        index=False,
    )
    print("Figure 3 top-1% visibility results were saved successfully.")


if __name__ == "__main__":
    
    figure3_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    cohort_folder = (
        "/Volumes/lydisk/work/work11/google_scholar/figure2/result/"
        "openalex/year_citation_dynamic"
    )

    citing_years = [
        1970, 1975, 1980, 1985,
        1990, 1995, 2000, 2005, 2007, 2009, 2010, 2012,
        2014, 2016, 2018, 2020,
    ]
    min_visibility_age = 1
    max_visibility_age = 20
    high_visibility_cutoff = 0.99
    closest_fraction = 0.25
    min_matched_references = 4
    save_paper_level = True

    calculate_top1_visibility_results(
        figure3_path,
        cohort_folder,
        citing_years,
        min_visibility_age,
        max_visibility_age,
        high_visibility_cutoff,
        closest_fraction,
        min_matched_references,
        save_paper_level,
    )
