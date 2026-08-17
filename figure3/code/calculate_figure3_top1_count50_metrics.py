import os

import numpy as np
import pandas as pd

import calculate_figure3_visibility_metrics as base


def calculate_equal_half_counts(matched, citing_year, min_matched_references):
    """Count top-1% references in equal closer and distant halves."""
    paper_keys = ["sample", "CitingPublicationId"]
    matched = matched.copy()
    matched["half_size"] = (
        matched["n_visibility_references"] // 2
    ).astype(int)
    matched["is_closer_half"] = (
        matched["similarity_rank"] <= matched["half_size"]
    )
    matched["is_distant_half"] = (
        matched["similarity_rank"]
        > matched["n_visibility_references"] - matched["half_size"]
    )

    summary = matched.groupby(paper_keys, sort=False).agg(
        n_visibility_references=("high_visibility", "size"),
        top1_reference_count=("high_visibility", "sum"),
        n_half_references=("half_size", "first"),
    )
    closer = (
        matched[matched["is_closer_half"]]
        .groupby(paper_keys)["high_visibility"]
        .sum()
        .rename("closer_top1_reference_count")
    )
    distant = (
        matched[matched["is_distant_half"]]
        .groupby(paper_keys)["high_visibility"]
        .sum()
        .rename("distant_top1_reference_count")
    )
    summary = summary.join(closer).join(distant)

    insufficient = summary["n_visibility_references"] < min_matched_references
    half_columns = [
        "closer_top1_reference_count",
        "distant_top1_reference_count",
    ]
    summary.loc[insufficient, half_columns] = np.nan
    summary["top1_reference_count_gap"] = (
        summary["distant_top1_reference_count"]
        - summary["closer_top1_reference_count"]
    )
    summary = summary.reset_index()
    summary["year"] = citing_year
    return summary


def calculate_count50_results(
    figure3_path,
    cohort_folder,
    citing_years,
    min_visibility_age,
    max_visibility_age,
    high_visibility_cutoff,
    min_matched_references,
):
    """Recalculate top-1% counts using equal semantic-proximity halves."""
    distance_folder = os.path.join(
        figure3_path,
        "result",
        "openalex",
        "year_topic_distance_embeding",
    )
    output_folder = os.path.join(figure3_path, "result", "visibility_top1")
    os.makedirs(output_folder, exist_ok=True)

    paper_parts = []
    quality_rows = []

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
        _, matched, _ = base.calculate_paper_metrics(
            edges,
            citing_year,
            visibility_lookup,
            min_visibility_age,
            max_visibility_age,
            closest_fraction=0.50,
            min_matched_references=min_matched_references,
        )
        paper_counts = calculate_equal_half_counts(
            matched,
            citing_year,
            min_matched_references,
        )
        paper_parts.append(paper_counts)
        quality_rows.extend(cohort_quality)

    papers = pd.concat(paper_parts, ignore_index=True)
    papers.to_csv(
        os.path.join(
            output_folder,
            "figure3_top1_count50_paper_metrics.csv.gz",
        ),
        index=False,
        compression="gzip",
    )
    pd.DataFrame(quality_rows).to_csv(
        os.path.join(
            output_folder,
            "figure3_top1_count50_cohort_quality.csv",
        ),
        index=False,
    )
    pd.DataFrame(
        [
            {
                "semantic_metric": "cosine_similarity",
                "high_visibility_percentile_cutoff": high_visibility_cutoff,
                "min_citation_age_at_citation": min_visibility_age,
                "max_citation_age_at_citation": max_visibility_age,
                "semantic_group_definition": (
                    "closest half and most distant half within each citing paper"
                ),
                "odd_reference_lists": "middle-ranked reference excluded",
                "min_matched_references": min_matched_references,
            }
        ]
    ).to_csv(
        os.path.join(
            output_folder,
            "figure3_top1_count50_parameters.csv",
        ),
        index=False,
    )
    print("Figure 3 equal-half count results were saved successfully.")


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
    min_matched_references = 4

    calculate_count50_results(
        figure3_path,
        cohort_folder,
        citing_years,
        min_visibility_age,
        max_visibility_age,
        high_visibility_cutoff,
        min_matched_references,
    )
