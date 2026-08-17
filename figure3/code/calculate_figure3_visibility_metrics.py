import os
import re
import glob
import numpy as np
import pandas as pd


def parse_sample(path):
    """Extract the sample number from a topic-distance file name."""
    match = re.search(r"_k(\d+)\.csv$", path)
    if match is None:
        return None
    return int(match.group(1))


def period_name(year):
    """Assign a citing year to one of the three study periods."""
    if year < 1990:
        return "Pre-digital (1970-1989)"
    if year < 2004:
        return "Digital publishing (1990-2003)"
    return "Google Scholar era (2004 onward)"


def t_multiplier(n):
    """Return the two-sided 95 percent t multiplier."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
    except ImportError:
        return 1.96 if n > 1 else np.nan


def load_year_edges(distance_folder, year):
    """Load the 30 cosine-similarity samples for one citing year."""
    frames = []
    pattern = os.path.join(distance_folder, f"topic_distance_{year}_k*.csv")
    for path in sorted(glob.glob(pattern)):
        sample = parse_sample(path)
        frame = pd.read_csv(
            path,
            usecols=[
                "CitingPublicationId",
                "CitingYear",
                "CitedPublicationId",
                "CitedYear",
                "TopicDistance",
            ],
        )
        frame["sample"] = sample
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No cosine-similarity files were found for {year}.")

    edges = pd.concat(frames, ignore_index=True)
    edges = edges.rename(columns={"TopicDistance": "cosine_similarity"})
    numeric_columns = [
        "CitingPublicationId",
        "CitedPublicationId",
        "CitedYear",
        "cosine_similarity",
    ]
    for column in numeric_columns:
        edges[column] = pd.to_numeric(edges[column], errors="coerce")
    edges = edges.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_columns)
    edges["CitedYear"] = edges["CitedYear"].astype(int)
    edges["citation_age_at_citation"] = year - edges["CitedYear"]
    return edges


def load_cohort_visibility(
    cohort_folder,
    publication_year,
    citation_age,
    needed_ids,
    high_visibility_cutoff,
):
    """Rank prior citations within one publication cohort using average ranks."""
    path = os.path.join(cohort_folder, f"{publication_year}_citation.csv")
    citation_columns = [f"C{age}" for age in range(citation_age)]
    if not os.path.exists(path):
        return pd.DataFrame(), {
            "publication_year": publication_year,
            "citation_age_at_citation": citation_age,
            "cohort_papers": 0,
            "needed_cited_papers": len(needed_ids),
            "matched_cited_papers": 0,
            "cohort_file_found": 0,
            "citation_history_complete": 0,
        }

    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing_columns = [
        column for column in citation_columns if column not in available_columns
    ]
    if missing_columns:
        return pd.DataFrame(), {
            "publication_year": publication_year,
            "citation_age_at_citation": citation_age,
            "cohort_papers": 0,
            "needed_cited_papers": len(needed_ids),
            "matched_cited_papers": 0,
            "cohort_file_found": 1,
            "citation_history_complete": 0,
        }

    cohort = pd.read_csv(path, usecols=["PaperID"] + citation_columns)
    cohort["PaperID"] = pd.to_numeric(cohort["PaperID"], errors="coerce")
    for column in citation_columns:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce").fillna(0)
    cohort["prior_citations"] = cohort[citation_columns].sum(axis=1)
    cohort = cohort.dropna(subset=["PaperID", "prior_citations"])
    cohort = cohort.drop_duplicates(subset=["PaperID"], keep="first")

    cohort_size = len(cohort)
    cohort["visibility_percentile"] = (
        cohort["prior_citations"].rank(method="average", ascending=True) - 0.5
    ) / cohort_size
    cohort["high_visibility"] = (
        cohort["visibility_percentile"] > high_visibility_cutoff
    ).astype(float)

    selected = cohort[cohort["PaperID"].isin(needed_ids)].copy()
    selected["publication_year"] = publication_year
    selected["citation_age_at_citation"] = citation_age
    selected = selected[
        [
            "PaperID",
            "publication_year",
            "citation_age_at_citation",
            "prior_citations",
            "visibility_percentile",
            "high_visibility",
        ]
    ]
    quality = {
        "publication_year": publication_year,
        "citation_age_at_citation": citation_age,
        "cohort_papers": cohort_size,
        "needed_cited_papers": len(needed_ids),
        "matched_cited_papers": len(selected),
        "cohort_file_found": 1,
        "citation_history_complete": 1,
    }
    return selected, quality


def build_visibility_lookup(
    edges,
    citing_year,
    cohort_folder,
    min_visibility_age,
    max_visibility_age,
    high_visibility_cutoff,
):
    """Build citation percentiles for every visibility-eligible cited paper."""
    lookup_parts = []
    quality_rows = []

    for citation_age in range(min_visibility_age, max_visibility_age + 1):
        publication_year = citing_year - citation_age
        selected_edges = edges[
            (edges["CitedYear"] == publication_year)
            & (edges["citation_age_at_citation"] == citation_age)
        ]
        needed_ids = selected_edges["CitedPublicationId"].dropna().unique()
        lookup, quality = load_cohort_visibility(
            cohort_folder,
            publication_year,
            citation_age,
            needed_ids,
            high_visibility_cutoff,
        )
        quality["citing_year"] = citing_year
        quality_rows.append(quality)
        if not lookup.empty:
            lookup_parts.append(lookup)

    if lookup_parts:
        lookup = pd.concat(lookup_parts, ignore_index=True)
    else:
        lookup = pd.DataFrame(
            columns=[
                "PaperID",
                "publication_year",
                "citation_age_at_citation",
                "prior_citations",
                "visibility_percentile",
                "high_visibility",
            ]
        )
    return lookup, quality_rows


def calculate_paper_metrics(
    edges,
    citing_year,
    visibility_lookup,
    min_visibility_age,
    max_visibility_age,
    closest_fraction,
    min_matched_references,
):
    """Calculate the three Figure 3 outcomes at the citing-paper level."""
    paper_keys = ["sample", "CitingPublicationId"]
    all_papers = edges.groupby(paper_keys, sort=False).agg(
        n_references_all=("cosine_similarity", "size"),
        mean_similarity_all=("cosine_similarity", "mean"),
    )

    eligible = edges[
        edges["citation_age_at_citation"].between(
            min_visibility_age,
            max_visibility_age,
        )
    ].copy()
    matched = eligible.merge(
        visibility_lookup,
        left_on=[
            "CitedPublicationId",
            "CitedYear",
            "citation_age_at_citation",
        ],
        right_on=[
            "PaperID",
            "publication_year",
            "citation_age_at_citation",
        ],
        how="inner",
        validate="many_to_one",
    )

    if matched.empty:
        papers = all_papers.reset_index()
        for column in [
            "n_visibility_references",
            "mean_similarity_matched",
            "high_visibility_share",
            "closest_high_visibility_share",
            "other_high_visibility_share",
            "high_visibility_gap",
        ]:
            papers[column] = np.nan
        papers["year"] = citing_year
        return papers, matched, eligible

    matched["similarity_rank"] = matched.groupby(paper_keys)[
        "cosine_similarity"
    ].rank(method="first", ascending=False)
    matched["n_visibility_references"] = matched.groupby(paper_keys)[
        "cosine_similarity"
    ].transform("size")
    matched["n_closest_references"] = np.ceil(
        closest_fraction * matched["n_visibility_references"]
    ).astype(int)
    matched["is_closest"] = (
        matched["similarity_rank"] <= matched["n_closest_references"]
    )

    matched_summary = matched.groupby(paper_keys, sort=False).agg(
        n_visibility_references=("cosine_similarity", "size"),
        mean_similarity_matched=("cosine_similarity", "mean"),
        high_visibility_share=("high_visibility", "mean"),
    )
    closest = matched[matched["is_closest"]].groupby(paper_keys)[
        "high_visibility"
    ].mean().rename("closest_high_visibility_share")
    other = matched[~matched["is_closest"]].groupby(paper_keys)[
        "high_visibility"
    ].mean().rename("other_high_visibility_share")
    matched_summary = matched_summary.join(closest).join(other)
    matched_summary["high_visibility_gap"] = (
        matched_summary["other_high_visibility_share"]
        - matched_summary["closest_high_visibility_share"]
    )

    insufficient = matched_summary["n_visibility_references"] < min_matched_references
    c_columns = [
        "closest_high_visibility_share",
        "other_high_visibility_share",
        "high_visibility_gap",
    ]
    matched_summary.loc[insufficient, c_columns] = np.nan

    papers = all_papers.join(matched_summary, how="left").reset_index()
    papers["year"] = citing_year
    return papers, matched, eligible


def calculate_sample_metrics(papers, edges, matched, eligible, citing_year):
    """Average citing-paper outcomes within each independently drawn sample."""
    metric_columns = [
        "mean_similarity_all",
        "mean_similarity_matched",
        "high_visibility_share",
        "closest_high_visibility_share",
        "other_high_visibility_share",
        "high_visibility_gap",
    ]
    rows = []
    samples = sorted(edges["sample"].dropna().unique())
    for sample in samples:
        paper_group = papers[papers["sample"] == sample]
        edge_group = edges[edges["sample"] == sample]
        eligible_group = eligible[eligible["sample"] == sample]
        matched_group = matched[matched["sample"] == sample]
        row = {
            "year": citing_year,
            "period": period_name(citing_year),
            "sample": int(sample),
            "n_edges_all": len(edge_group),
            "n_edges_visibility_eligible": len(eligible_group),
            "n_edges_visibility_matched": len(matched_group),
            "edge_match_rate_among_eligible": (
                len(matched_group) / len(eligible_group)
                if len(eligible_group) else np.nan
            ),
            "n_papers_all": len(paper_group),
            "n_papers_visibility_matched": int(
                paper_group["high_visibility_share"].notna().sum()
            ),
            "n_papers_closest_analysis": int(
                paper_group["high_visibility_gap"].notna().sum()
            ),
        }
        for metric in metric_columns:
            row[metric] = paper_group[metric].mean()
        rows.append(row)
    return rows


def summarize_years(samples):
    """Calculate yearly means and sample-based 95 percent confidence intervals."""
    metric_columns = [
        "mean_similarity_all",
        "mean_similarity_matched",
        "high_visibility_share",
        "closest_high_visibility_share",
        "other_high_visibility_share",
        "high_visibility_gap",
    ]
    rows = []
    for year, group in samples.groupby("year"):
        row = {
            "year": int(year),
            "period": period_name(int(year)),
            "n_samples": group["sample"].nunique(),
            "n_edges_all": group["n_edges_all"].sum(),
            "n_edges_visibility_eligible": group["n_edges_visibility_eligible"].sum(),
            "n_edges_visibility_matched": group["n_edges_visibility_matched"].sum(),
            "edge_match_rate_among_eligible": (
                group["n_edges_visibility_matched"].sum()
                / group["n_edges_visibility_eligible"].sum()
                if group["n_edges_visibility_eligible"].sum() else np.nan
            ),
            "n_papers_all": group["n_papers_all"].sum(),
            "n_papers_visibility_matched": group["n_papers_visibility_matched"].sum(),
            "n_papers_closest_analysis": group["n_papers_closest_analysis"].sum(),
        }
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            se = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan
            half_width = t_multiplier(n) * se if n > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_low"] = mean - half_width
            row[f"{metric}_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def summarize_periods(years):
    """Give each observed year equal weight when summarizing the three periods."""
    metric_columns = [
        "mean_similarity_all",
        "mean_similarity_matched",
        "high_visibility_share",
        "closest_high_visibility_share",
        "other_high_visibility_share",
        "high_visibility_gap",
    ]
    rows = []
    for period, group in years.groupby("period", sort=False):
        row = {"period": period, "n_years": len(group)}
        for metric in metric_columns:
            values = group[f"{metric}_mean"].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            se = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan
            half_width = t_multiplier(n) * se if n > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_low"] = mean - half_width
            row[f"{metric}_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    figure3_path = "/Volumes/lydisk/work/work11/google_scholar/figure3"
    distance_folder = os.path.join(
        figure3_path,
        "result",
        "openalex",
        "year_topic_distance_embeding",
    )
    cohort_folder = (
        "/Volumes/lydisk/work/work11/google_scholar/figure2/result/"
        "openalex/year_citation_dynamic"
    )
    output_folder = os.path.join(figure3_path, "result", "visibility")

    citing_years = [
        1970, 1975, 1980, 1985,
        1990, 1995, 2000, 2005, 2007, 2009, 2010, 2012,
        2014, 2016, 2018, 2020,
    ]
    min_visibility_age = 1
    max_visibility_age = 20
    high_visibility_cutoff = 0.80
    closest_fraction = 0.25
    min_matched_references = 4
    save_paper_level = True

    os.makedirs(output_folder, exist_ok=True)

    paper_parts = []
    sample_rows = []
    cohort_quality_rows = []

    for citing_year in citing_years:
        print(f"Processing citing year {citing_year}")
        edges = load_year_edges(distance_folder, citing_year)
        visibility_lookup, cohort_quality = build_visibility_lookup(
            edges,
            citing_year,
            cohort_folder,
            min_visibility_age,
            max_visibility_age,
            high_visibility_cutoff,
        )
        papers, matched, eligible = calculate_paper_metrics(
            edges,
            citing_year,
            visibility_lookup,
            min_visibility_age,
            max_visibility_age,
            closest_fraction,
            min_matched_references,
        )
        sample_rows.extend(
            calculate_sample_metrics(papers, edges, matched, eligible, citing_year)
        )
        cohort_quality_rows.extend(cohort_quality)
        if save_paper_level:
            paper_parts.append(papers)

    samples = pd.DataFrame(sample_rows).sort_values(["year", "sample"])
    years = summarize_years(samples)
    periods = summarize_periods(years)
    cohort_quality = pd.DataFrame(cohort_quality_rows).sort_values(
        ["citing_year", "citation_age_at_citation"]
    )

    samples.to_csv(
        os.path.join(output_folder, "figure3_visibility_sample_metrics.csv"),
        index=False,
    )
    years.to_csv(
        os.path.join(output_folder, "figure3_visibility_year_metrics.csv"),
        index=False,
    )
    periods.to_csv(
        os.path.join(output_folder, "figure3_visibility_period_metrics.csv"),
        index=False,
    )
    cohort_quality.to_csv(
        os.path.join(output_folder, "figure3_visibility_cohort_quality.csv"),
        index=False,
    )

    if save_paper_level:
        papers = pd.concat(paper_parts, ignore_index=True)
        papers.to_csv(
            os.path.join(output_folder, "figure3_visibility_paper_metrics.csv.gz"),
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
                "high_visibility_definition": "visibility percentile above cutoff",
                "closest_reference_fraction": closest_fraction,
                "min_matched_references_for_closest_analysis": min_matched_references,
                "paper_weighting": "equal",
                "sample_weighting_within_year": "equal",
                "year_weighting_within_period": "equal",
            }
        ]
    )
    parameters.to_csv(
        os.path.join(output_folder, "figure3_visibility_parameters.csv"),
        index=False,
    )
    print("Figure 3 visibility results were saved successfully.")
