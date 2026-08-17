#!/usr/bin/env python3
"""Calculate cross-field citation metrics for Figure 4."""

import glob
import math
import os
import sqlite3
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=64)
def field_map(concept_path, year):
    """Load the highest-scoring level-0 field for each paper in one year."""
    input_file = os.path.join(concept_path, f"pub_concept_{year}.db")
    if not os.path.exists(input_file):
        print(f"[Warning] Missing concept database: {input_file}")
        return {}

    connection = sqlite3.connect(input_file)
    best_fields = {}
    query = (
        "SELECT PublicationId, FieldId, Score "
        "FROM pub_concept WHERE FieldLevel=0"
    )

    for paper_id, field_id, score in connection.execute(query):
        try:
            paper_id = int(float(paper_id))
            field_id = int(field_id)
            score = float(score)
        except (TypeError, ValueError):
            continue

        current = best_fields.get(paper_id)
        if current is None or score > current[1]:
            best_fields[paper_id] = (field_id, score)

    connection.close()
    return {
        paper_id: field_score[0]
        for paper_id, field_score in best_fields.items()
    }


def gini(values):
    """Calculate the Gini coefficient for non-negative values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan
    if np.any(values < 0):
        raise ValueError("Values used for Gini must be non-negative.")

    total = values.sum()
    if total == 0:
        return 0.0

    values.sort()
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return (
        2.0 * np.dot(ranks, values) / (values.size * total)
        - (values.size + 1.0) / values.size
    )


def expected_distinct(field_counts, sample_size):
    """Calculate the expected number of fields after citation rarefaction."""
    total = sum(field_counts)
    if total < sample_size:
        return np.nan

    log_denominator = (
        math.lgamma(total + 1)
        - math.lgamma(sample_size + 1)
        - math.lgamma(total - sample_size + 1)
    )
    expected_absent = 0.0

    for count in field_counts:
        remaining = total - count
        if remaining >= sample_size:
            log_numerator = (
                math.lgamma(remaining + 1)
                - math.lgamma(sample_size + 1)
                - math.lgamma(remaining - sample_size + 1)
            )
            expected_absent += math.exp(log_numerator - log_denominator)

    return len(field_counts) - expected_absent


def rarefied_entropy(field_counts, sample_size, random_generator, draws):
    """Estimate rarefied Shannon entropy by sampling without replacement."""
    total = sum(field_counts)
    if total < sample_size:
        return np.nan

    field_pool = np.repeat(np.arange(len(field_counts)), field_counts)
    entropy_values = np.empty(draws, dtype=float)

    for draw in range(draws):
        selected = random_generator.choice(
            field_pool,
            size=sample_size,
            replace=False,
        )
        _, counts = np.unique(selected, return_counts=True)
        probabilities = counts / sample_size
        entropy_values[draw] = -(
            probabilities * np.log2(probabilities)
        ).sum()

    return entropy_values.mean()


def load_cohort_strata(
    citation_path,
    year,
    citation_column,
    percentile_groups,
):
    """Assign citation strata using the complete Figure 1 cohort table."""
    input_file = os.path.join(citation_path, f"citation_year{year}.csv")
    if not os.path.exists(input_file):
        print(f"[Warning] Missing citation cohort: {input_file}")
        return {}, {}, 0

    data = pd.read_csv(input_file, usecols=["PaperID", citation_column])
    data["PaperID"] = pd.to_numeric(data["PaperID"], errors="coerce")
    data[citation_column] = pd.to_numeric(
        data[citation_column], errors="coerce"
    )
    data = data.dropna(subset=["PaperID", citation_column]).copy()

    if data["PaperID"].duplicated().any():
        raise ValueError(f"Duplicate PaperID values in {input_file}")
    if (data[citation_column] < 0).any():
        raise ValueError(f"Negative citation counts in {input_file}")

    data["PaperID"] = data["PaperID"].astype(np.int64)
    data = data.sort_values(
        [citation_column, "PaperID"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    labels = np.full(len(data), "", dtype=object)
    group_sizes = {}

    for group_name, lower, upper in percentile_groups:
        start = int(np.floor(len(data) * lower))
        stop = int(np.floor(len(data) * upper))
        labels[start:stop] = group_name
        group_sizes[group_name] = stop - start

    data["stratum"] = labels
    selected = data[data["stratum"] != ""]
    stratum_map = dict(zip(selected["PaperID"], selected["stratum"]))
    group_sizes["ALL(top15%)"] = sum(group_sizes.values())
    return stratum_map, group_sizes, len(data)


def build_paper_table(
    year,
    edge_path,
    concept_path,
    target_papers,
    citation_age_min,
    citation_age_max,
    expected_shards,
):
    """Build field-specific citation counts for one publication cohort."""
    cited_fields = field_map(concept_path, year)
    shard_files = sorted(
        glob.glob(
            os.path.join(
                edge_path,
                str(year),
                f"edgelist_year{year}_*.csv",
            )
        )
    )

    if expected_shards is not None and len(shard_files) != expected_shards:
        print(
            f"[Warning] {year} contains {len(shard_files)} shard files; "
            f"expected {expected_shards}."
        )

    paper_counts = {}
    cited_papers_in_window = set()
    n_edges_raw = 0
    n_edges_window = 0
    n_edges_top15 = 0
    n_citing_field_known = 0
    n_cited_field_known = 0

    for input_file in shard_files:
        shard = int(input_file.rsplit("_", 1)[1].split(".")[0])
        columns = [
            "CitingPublicationId",
            "CitedPublicationId",
            "CitingYear",
            "CitedYear",
        ]
        data = pd.read_csv(input_file, usecols=columns)
        n_edges_raw += len(data)

        for column in columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=columns).copy()
        data = data[data["CitedYear"] == year]

        citation_age = data["CitingYear"] - year
        data = data[
            (citation_age >= citation_age_min)
            & (citation_age <= citation_age_max)
        ].copy()
        n_edges_window += len(data)
        if data.empty:
            continue

        data["citing_id"] = data["CitingPublicationId"].astype(np.int64)
        data["cited_id"] = data["CitedPublicationId"].astype(np.int64)
        data = data[data["cited_id"].isin(target_papers)].copy()
        n_edges_top15 += len(data)
        if data.empty:
            continue

        cited_papers_in_window.update(data["cited_id"].unique().tolist())

        citing_field_values = np.empty(len(data), dtype=object)
        citing_years = data["CitingYear"].astype(int).to_numpy()
        citing_ids = data["citing_id"].to_numpy()

        for citing_year in np.unique(citing_years):
            citing_fields = field_map(concept_path, int(citing_year))
            selected = citing_years == citing_year
            citing_field_values[selected] = [
                citing_fields.get(paper_id)
                for paper_id in citing_ids[selected]
            ]

        data["citing_field"] = citing_field_values
        data["cited_field"] = data["cited_id"].map(cited_fields)
        n_citing_field_known += int(data["citing_field"].notna().sum())
        n_cited_field_known += int(data["cited_field"].notna().sum())

        complete = data[
            data["citing_field"].notna()
            & data["cited_field"].notna()
        ]

        for cited_id, group in complete.groupby("cited_id"):
            field_counts = Counter(group["citing_field"].tolist())
            cited_field = int(group["cited_field"].iloc[0])
            cited_id = int(cited_id)

            if cited_id not in paper_counts:
                paper_counts[cited_id] = {
                    "cited_field": cited_field,
                    "shard": shard,
                    "field_counts": field_counts,
                }
            else:
                if paper_counts[cited_id]["shard"] != shard:
                    raise ValueError(
                        f"Paper {cited_id} appears in multiple shards."
                    )
                paper_counts[cited_id]["field_counts"].update(field_counts)

    rows = [
        (
            cited_id,
            values["cited_field"],
            values["shard"],
            dict(values["field_counts"]),
        )
        for cited_id, values in paper_counts.items()
    ]
    coverage = {
        "n_edges_raw": n_edges_raw,
        "n_edges_window": n_edges_window,
        "n_edges_top15": n_edges_top15,
        "n_citing_field_known": n_citing_field_known,
        "n_cited_field_known": n_cited_field_known,
        "n_cited_papers_window": len(cited_papers_in_window),
        "n_papers_field_ok": len(rows),
    }
    return rows, coverage


def empirical_interval(values):
    """Return the 2.5% and 97.5% quantiles of shard estimates."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return np.quantile(values, [0.025, 0.975])


def cohort_metrics(
    year,
    edge_path,
    concept_path,
    citation_path,
    citation_column,
    citation_age_min,
    citation_age_max,
    percentile_groups,
    rarefaction_sizes,
    rarefaction_draws,
    expected_shards,
    random_generator,
):
    """Calculate all Figure 4 metrics for one publication cohort."""
    stratum_map, stratum_sizes, n_cohort_papers = load_cohort_strata(
        citation_path,
        year,
        citation_column,
        percentile_groups,
    )
    if not stratum_map:
        return None

    rows, coverage = build_paper_table(
        year,
        edge_path,
        concept_path,
        set(stratum_map),
        citation_age_min,
        citation_age_max,
        expected_shards,
    )
    if not rows:
        return None

    cited_ids = np.array([row[0] for row in rows], dtype=np.int64)
    cited_fields = np.array([row[1] for row in rows], dtype=np.int64)
    shards = np.array([row[2] for row in rows], dtype=np.int16)
    counters = [row[3] for row in rows]

    total = np.array(
        [sum(counter.values()) for counter in counters],
        dtype=np.int64,
    )
    same = np.array(
        [
            counter.get(field_id, 0)
            for counter, field_id in zip(counters, cited_fields)
        ],
        dtype=np.int64,
    )
    cross = total - same
    cross_share = np.where(total > 0, cross / total, np.nan)
    strata = np.array(
        [stratum_map.get(paper_id, "") for paper_id in cited_ids],
        dtype=object,
    )

    crossfield_rows = []
    diversity_rows = []
    gini_rows = []
    stratum_names = [group[0] for group in percentile_groups]
    stratum_names.append("ALL(top15%)")

    for stratum_name in stratum_names:
        if stratum_name == "ALL(top15%)":
            selected = strata != ""
        else:
            selected = strata == stratum_name

        n_ranked = int(stratum_sizes.get(stratum_name, 0))
        n_field_eligible = int(selected.sum())
        if n_field_eligible == 0:
            continue

        field_coverage = (
            n_field_eligible / n_ranked if n_ranked > 0 else np.nan
        )
        selected_shards = np.unique(shards[selected])

        shard_means = []
        for shard in selected_shards:
            mask = (
                selected
                & (shards == shard)
                & np.isfinite(cross_share)
            )
            if mask.sum() > 0:
                shard_means.append(np.nanmean(cross_share[mask]))
        share_ci_low, share_ci_high = empirical_interval(shard_means)
        crossfield_rows.append({
            "cohort": year,
            "stratum": stratum_name,
            "mean": float(np.nanmean(cross_share[selected])),
            "ci_lo": float(share_ci_low),
            "ci_hi": float(share_ci_high),
            "n_ranked": n_ranked,
            "n_papers": n_field_eligible,
            "field_coverage": field_coverage,
        })

        cross_gini_shards = []
        same_gini_shards = []
        for shard in selected_shards:
            mask = selected & (shards == shard)
            if mask.sum() >= 2:
                cross_gini_shards.append(gini(cross[mask]))
                same_gini_shards.append(gini(same[mask]))
        cross_ci_low, cross_ci_high = empirical_interval(
            cross_gini_shards
        )
        same_ci_low, same_ci_high = empirical_interval(same_gini_shards)
        gini_rows.append({
            "cohort": year,
            "stratum": stratum_name,
            "gini_cross": float(gini(cross[selected])),
            "cross_ci_lo": float(cross_ci_low),
            "cross_ci_hi": float(cross_ci_high),
            "gini_same": float(gini(same[selected])),
            "same_ci_lo": float(same_ci_low),
            "same_ci_hi": float(same_ci_high),
            "mean_cross_count": float(np.mean(cross[selected])),
            "mean_same_count": float(np.mean(same[selected])),
            "median_cross_count": float(np.median(cross[selected])),
            "median_same_count": float(np.median(same[selected])),
            "n_ranked": n_ranked,
            "n_papers": n_field_eligible,
            "field_coverage": field_coverage,
        })

        for sample_size in rarefaction_sizes:
            eligible = selected & (total >= sample_size)
            if eligible.sum() == 0:
                continue

            indices = np.flatnonzero(eligible)
            distinct = np.array([
                expected_distinct(
                    list(counters[index].values()),
                    sample_size,
                )
                for index in indices
            ])
            entropy = np.array([
                rarefied_entropy(
                    list(counters[index].values()),
                    sample_size,
                    random_generator,
                    rarefaction_draws,
                )
                for index in indices
            ])
            eligible_shards = shards[indices]
            distinct_shards = []
            entropy_shards = []

            for shard in np.unique(eligible_shards):
                mask = eligible_shards == shard
                distinct_shards.append(np.nanmean(distinct[mask]))
                entropy_shards.append(np.nanmean(entropy[mask]))

            distinct_ci_low, distinct_ci_high = empirical_interval(
                distinct_shards
            )
            entropy_ci_low, entropy_ci_high = empirical_interval(
                entropy_shards
            )
            diversity_rows.append({
                "cohort": year,
                "stratum": stratum_name,
                "n_rarefy": sample_size,
                "mean_distinct": float(np.nanmean(distinct)),
                "distinct_ci_lo": float(distinct_ci_low),
                "distinct_ci_hi": float(distinct_ci_high),
                "mean_entropy": float(np.nanmean(entropy)),
                "entropy_ci_lo": float(entropy_ci_low),
                "entropy_ci_hi": float(entropy_ci_high),
                "n_ranked": n_ranked,
                "n_eligible": int(eligible.sum()),
                "field_coverage": field_coverage,
            })

    n_top15_ranked = int(stratum_sizes.get("ALL(top15%)", 0))
    n_top15_field_ok = int(np.count_nonzero(strata != ""))
    coverage_row = {
        "cohort": year,
        **coverage,
        "n_cohort_papers": n_cohort_papers,
        "n_top15_ranked": n_top15_ranked,
        "n_top15_field_ok": n_top15_field_ok,
        "top15_field_coverage": (
            n_top15_field_ok / n_top15_ranked
            if n_top15_ranked > 0
            else np.nan
        ),
        "citing_field_known_rate": (
            coverage["n_citing_field_known"]
            / max(coverage["n_edges_top15"], 1)
        ),
        "cited_field_known_rate": (
            coverage["n_cited_field_known"]
            / max(coverage["n_edges_top15"], 1)
        ),
    }

    print(
        f"[{year}] cohort={n_cohort_papers:,}  "
        f"top15_field_coverage="
        f"{coverage_row['top15_field_coverage']:.3f}  "
        f"edges_window={coverage['n_edges_window']:,}",
        flush=True,
    )
    return (
        crossfield_rows,
        diversity_rows,
        gini_rows,
        coverage_row,
    )


def calculate_all_cohorts(
    edge_path,
    concept_path,
    citation_path,
    output_path,
    start_year,
    end_year,
    citation_column,
    citation_age_min,
    citation_age_max,
    percentile_groups,
    rarefaction_sizes,
    rarefaction_draws,
    expected_shards,
    random_seed,
):
    """Calculate and save Figure 4 metrics for all available cohorts."""
    os.makedirs(output_path, exist_ok=True)
    random_generator = np.random.default_rng(random_seed)

    crossfield_results = []
    diversity_results = []
    gini_results = []
    coverage_results = []
    
    for year in range(start_year, end_year + 1):
        cohort_folder = os.path.join(edge_path, str(year))
        if not os.path.isdir(cohort_folder):
            print(f"[Warning] Missing edge folder for {year}; skipped.")
            continue

        results = cohort_metrics(
            year,
            edge_path,
            concept_path,
            citation_path,
            citation_column,
            citation_age_min,
            citation_age_max,
            percentile_groups,
            rarefaction_sizes,
            rarefaction_draws,
            expected_shards,
            random_generator,
        )
        if results is None:
            continue

        crossfield_rows, diversity_rows, gini_rows, coverage_row = results
        crossfield_results.extend(crossfield_rows)
        diversity_results.extend(diversity_rows)
        gini_results.extend(gini_rows)
        coverage_results.append(coverage_row)

    pd.DataFrame(crossfield_results).to_csv(
        os.path.join(output_path, "fig4_a_crossfield_share.csv"),
        index=False,
    )
    pd.DataFrame(diversity_results).to_csv(
        os.path.join(output_path, "fig4_b_diversity.csv"),
        index=False,
    )
    pd.DataFrame(gini_results).to_csv(
        os.path.join(output_path, "fig4_c_gini.csv"),
        index=False,
    )
    pd.DataFrame(coverage_results).to_csv(
        os.path.join(output_path, "fig4_coverage_log.csv"),
        index=False,
    )
    print(f"All Figure 4 results were saved in: {output_path}")


if __name__ == "__main__":

    base_path = "/path/on/server/figure4/result/openalex"
    citation_path = (
        "/path/on/server/figure1/result/openalex/year_citation_journal"
    )
    edge_path = os.path.join(base_path, "year_edgelist_sub")
    concept_path = os.path.join(base_path, "year_pub_concept")
    output_path = os.path.join(base_path, "fig4_metrics")

    start_year = 1950
    end_year = 2020
    citation_column = "C5"
    citation_age_min = 0
    citation_age_max = 5

    percentile_groups = [
        ("top1%", 0.00, 0.01),
        ("1-5%", 0.01, 0.05),
        ("5-10%", 0.05, 0.10),
        ("10-15%", 0.10, 0.15),
    ]
    rarefaction_sizes = [5, 10, 20]
    rarefaction_draws = 50
    expected_shards = 49
    random_seed = 20260812
    
    calculate_all_cohorts(
        edge_path,
        concept_path,
        citation_path,
        output_path,
        start_year,
        end_year,
        citation_column,
        citation_age_min,
        citation_age_max,
        percentile_groups,
        rarefaction_sizes,
        rarefaction_draws,
        expected_shards,
        random_seed,
    )
    