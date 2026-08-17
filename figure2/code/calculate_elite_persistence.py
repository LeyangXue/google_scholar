#!/usr/bin/env python3
"""Calculate cohort-based elite retention and survival statistics.

For each publication-year cohort, annual citation increments C0...Cn are
converted to cumulative citations A(a) = sum(C0...Ca). Elite papers are all
papers at or above the citation threshold containing the top fraction of the
cohort; all papers tied at the threshold are retained.

Two-year retention compares elite membership at citation ages 2 and 4.
Five-year retention compares elite membership at citation ages 5 and 10.
Survival starts among papers elite at age 2 and ends at their first subsequent
exit from the elite, with papers still elite at the final observed age treated
as right-censored.
"""

import csv
import glob
import math
import os

import numpy as np
import pandas as pd


def period_name(year):
    """Return the historical period for a publication cohort."""
    if year <= 1989:
        return "Pre-digital (1950-1989)"
    if year <= 2003:
        return "Digital publishing (1990-2003)"
    return "Google Scholar era (2004+)"


def citation_columns(columns):
    """Return continuous citation-age columns ordered from C0 onward."""
    pairs = []
    for column in columns:
        if column.startswith("C") and column[1:].isdigit():
            pairs.append((int(column[1:]), column))
    pairs.sort()

    ages = [age for age, _ in pairs]
    if not ages or ages != list(range(ages[-1] + 1)):
        raise ValueError("Citation-age columns must be continuous from C0.")
    return [column for _, column in pairs]


def elite_threshold(values, top_fraction):
    """Return the tie-inclusive elite threshold and membership information."""
    n_papers = values.size
    target_size = max(1, int(math.ceil(n_papers * top_fraction)))
    cutoff_index = n_papers - target_size
    threshold = np.partition(values, cutoff_index)[cutoff_index]
    elite_mask = values >= threshold
    actual_size = int(np.count_nonzero(elite_mask))
    return threshold, elite_mask, target_size, actual_size


def validate_cohort(data, citation_cols, publication_year, input_file):
    """Validate identifiers, years, and citation increments for one cohort."""
    if data.empty:
        raise ValueError(f"Empty cohort file: {input_file}")
    if data["PaperID"].isna().any():
        raise ValueError(f"Missing PaperID values: {input_file}")
    if data["PaperID"].duplicated().any():
        raise ValueError(f"Duplicate PaperID values: {input_file}")
    if not (data["year"] == publication_year).all():
        raise ValueError(f"Year values do not match filename: {input_file}")
    if data[citation_cols].isna().any().any():
        raise ValueError(f"Missing citation values: {input_file}")
    if (data[citation_cols] < 0).any().any():
        raise ValueError(f"Negative citation values: {input_file}")


def write_survival_records(
    output_file,
    dataset_name,
    publication_year,
    paper_ids,
    followup_time,
    exit_age,
    event,
    last_observed_age,
    write_header,
):
    """Append paper-level elite-duration records to a dataset CSV file."""
    records = pd.DataFrame({
        "Dataset": dataset_name,
        "PaperID": paper_ids,
        "Publication_year": publication_year,
        "Period": period_name(publication_year),
        "Baseline_age": 2,
        "Followup_time": followup_time,
        "Exit_age": exit_age,
        "Event": event.astype(np.int8),
        "Last_observed_age": last_observed_age,
    })
    records.to_csv(
        output_file,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def process_cohort(
    input_file,
    dataset_name,
    publication_year,
    top_fraction,
    survival_output_file,
    write_survival_header,
):
    """Calculate thresholds, retention, and paper-level survival for one cohort."""
    header = pd.read_csv(input_file, nrows=0).columns.tolist()
    citation_cols = citation_columns(header)
    max_age = len(citation_cols) - 1
    
    dtype_map = {"PaperID": np.float64, "year": np.int16}
    dtype_map.update({column: np.int32 for column in citation_cols})
    data = pd.read_csv(
        input_file,
        usecols=["PaperID", "year"] + citation_cols,
        dtype=dtype_map,
    )
    validate_cohort(data, citation_cols, publication_year, input_file)

    paper_ids = data["PaperID"].to_numpy()
    if not np.all(np.isfinite(paper_ids)) or not np.all(paper_ids == np.floor(paper_ids)):
        raise ValueError(f"PaperID values are not finite integers: {input_file}")
    paper_ids = paper_ids.astype(np.int64)

    citations = data[citation_cols].to_numpy(dtype=np.int32, copy=False)
    n_papers = len(data)
    cumulative = np.zeros(n_papers, dtype=np.int64)
    saved_elite_masks = {}
    threshold_rows = []

    baseline_indices = None
    alive = None
    followup_time = None
    exit_age = None
    event = None

    for age in range(max_age + 1):
        cumulative += citations[:, age]

        if age < 2:
            continue

        threshold, elite_mask, target_size, actual_size = elite_threshold(
            cumulative, top_fraction
        )
        threshold_rows.append({
            "Dataset": dataset_name,
            "Publication_year": publication_year,
            "Citation_age": age,
            "N_papers": n_papers,
            "Top_fraction": top_fraction,
            "Target_elite_size": target_size,
            "Citation_threshold": int(threshold),
            "Actual_elite_size": actual_size,
            "Actual_elite_fraction": actual_size / n_papers,
        })

        if age in (2, 4, 5, 10):
            saved_elite_masks[age] = elite_mask.copy()

        if age == 2:
            baseline_indices = np.flatnonzero(elite_mask)
            n_baseline = baseline_indices.size
            alive = np.ones(n_baseline, dtype=bool)
            followup_time = np.full(n_baseline, max_age - 2, dtype=np.int16)
            exit_age = np.full(n_baseline, -1, dtype=np.int16)
            event = np.zeros(n_baseline, dtype=np.int8)
        else:
            still_elite = elite_mask[baseline_indices]
            newly_exited = alive & ~still_elite
            if np.any(newly_exited):
                followup_time[newly_exited] = age - 2
                exit_age[newly_exited] = age
                event[newly_exited] = 1
            alive &= still_elite

    retention_rows = []
    retention_definitions = [
        ("Two-year forward retention", 2, 4),
        ("Five-year forward retention", 5, 10),
    ]
    for measure, baseline_age, followup_age in retention_definitions:
        if followup_age not in saved_elite_masks:
            continue
        baseline_mask = saved_elite_masks[baseline_age]
        followup_mask = saved_elite_masks[followup_age]
        baseline_size = int(np.count_nonzero(baseline_mask))
        followup_size = int(np.count_nonzero(followup_mask))
        overlap_size = int(np.count_nonzero(baseline_mask & followup_mask))
        retention_rows.append({
            "Dataset": dataset_name,
            "Publication_year": publication_year,
            "Period": period_name(publication_year),
            "Measure": measure,
            "Baseline_age": baseline_age,
            "Followup_age": followup_age,
            "N_papers": n_papers,
            "Baseline_elite_size": baseline_size,
            "Followup_elite_size": followup_size,
            "Overlap_size": overlap_size,
            "Retention": overlap_size / baseline_size,
        })

    if baseline_indices is not None:
        write_survival_records(
            survival_output_file,
            dataset_name,
            publication_year,
            paper_ids[baseline_indices],
            followup_time,
            exit_age,
            event,
            max_age,
            write_survival_header,
        )
        write_survival_header = False

    del data, citations, cumulative
    return retention_rows, threshold_rows, write_survival_header


def moving_block_bootstrap(values, iterations, block_length, rng):
    """Return a moving-block bootstrap distribution of the mean."""
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
    retention_data,
    bootstrap_iterations,
    bootstrap_block_length,
    random_seed,
):
    """Calculate period means, confidence intervals, and within-period slopes."""
    rng = np.random.default_rng(random_seed)
    rows = []

    for (dataset, measure, period), group in retention_data.groupby(
        ["Dataset", "Measure", "Period"], sort=False
    ):
        group = group.sort_values("Publication_year")
        years = group["Publication_year"].to_numpy(dtype=float)
        values = group["Retention"].to_numpy(dtype=float)
        bootstrap = moving_block_bootstrap(
            values,
            bootstrap_iterations,
            bootstrap_block_length,
            rng,
        )
        slope = np.polyfit(years, values, 1)[0] if len(values) >= 2 else np.nan
        rows.append({
            "Dataset": dataset,
            "Measure": measure,
            "Period": period,
            "Start_year": int(years.min()),
            "End_year": int(years.max()),
            "N_years": len(values),
            "Mean_retention": values.mean(),
            "CI_lower": np.quantile(bootstrap, 0.025),
            "CI_upper": np.quantile(bootstrap, 0.975),
            "Slope_per_year": slope,
        })
    return pd.DataFrame(rows)


def calculate_segmented_slopes(retention_data, breakpoint_year):
    """Estimate pre/post-breakpoint slopes and the change in slope."""
    rows = []

    for (dataset, measure), group in retention_data.groupby(
        ["Dataset", "Measure"], sort=False
    ):
        group = group.sort_values("Publication_year")
        years = group["Publication_year"].to_numpy(dtype=float)
        response = group["Retention"].to_numpy(dtype=float)
        centered_year = years - breakpoint_year
        post = (years >= breakpoint_year).astype(float)
        design = np.column_stack([
            np.ones(len(years)),
            centered_year,
            post,
            centered_year * post,
        ])
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        rows.append({
            "Dataset": dataset,
            "Measure": measure,
            "Breakpoint_year": breakpoint_year,
            "N_years": len(years),
            "Pre_break_slope": coefficients[1],
            "Level_change_at_break": coefficients[2],
            "Slope_change": coefficients[3],
            "Post_break_slope": coefficients[1] + coefficients[3],
        })
    return pd.DataFrame(rows)


def calculate_kaplan_meier(paper_level_file):
    """Aggregate paper-level durations into period-specific Kaplan-Meier curves."""
    data = pd.read_csv(
        paper_level_file,
        usecols=["Dataset", "Period", "Followup_time", "Event"],
    )
    rows = []

    for (dataset, period), group in data.groupby(["Dataset", "Period"], sort=False):
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
                "Period": period,
                "Time_since_age_2": time,
                "At_risk": at_risk,
                "Events": events,
                "Censored": censored,
                "Survival": survival,
                "Standard_error": standard_error,
                "CI_lower": max(0.0, survival - 1.96 * standard_error),
                "CI_upper": min(1.0, survival + 1.96 * standard_error),
            })
    return pd.DataFrame(rows)


def calculate_dataset(
    data_path,
    dataset_name,
    output_path,
    start_year,
    end_year,
    top_fraction,
):
    """Process all available publication cohorts for one dataset."""
    input_files = sorted(glob.glob(os.path.join(data_path, "*_citation.csv")))
    input_files = [
        input_file for input_file in input_files
        if start_year <= int(os.path.basename(input_file).split("_")[0]) <= end_year
    ]
    if not input_files:
        raise FileNotFoundError(f"No cohort files found in {data_path}")

    survival_output_file = os.path.join(
        output_path, f"elite_survival_paper_level_{dataset_name.lower()}.csv"
    )
    retention_rows = []
    threshold_rows = []
    write_survival_header = True

    for file_number, input_file in enumerate(input_files, start=1):
        publication_year = int(os.path.basename(input_file).split("_")[0])
        print(
            f"[{dataset_name}] {file_number}/{len(input_files)}: "
            f"processing publication year {publication_year}",
            flush=True,
        )
        cohort_retention, cohort_thresholds, write_survival_header = process_cohort(
            input_file,
            dataset_name,
            publication_year,
            top_fraction,
            survival_output_file,
            write_survival_header,
        )
        retention_rows.extend(cohort_retention)
        threshold_rows.extend(cohort_thresholds)

    return (
        pd.DataFrame(retention_rows),
        pd.DataFrame(threshold_rows),
        survival_output_file,
    )


if __name__ == "__main__":
    
    mag_data_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/result/mag/year_citation_dynamic"
    openalex_data_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/result/openalex/year_citation_dynamic"
    output_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/result/retention_survival"

    start_year = 1950
    end_year = 2023
    top_fraction = 0.01
    bootstrap_iterations = 2000
    bootstrap_block_length = 5
    random_seed = 20260712
    segmented_breakpoint_year = 2004
    
    os.makedirs(output_path, exist_ok=True)

    dataset_paths = {
        "MAG": mag_data_path,
        "OpenAlex": openalex_data_path,
    }

    all_retention = []
    all_thresholds = []
    survival_files = []

    for dataset_name, data_path in dataset_paths.items():
        retention, thresholds, survival_file = calculate_dataset(
            data_path,
            dataset_name,
            output_path,
            start_year,
            end_year,
            top_fraction,
        )
        retention.to_csv(
            os.path.join(
                output_path,
                f"elite_retention_yearly_{dataset_name.lower()}.csv",
            ),
            index=False,
        )
        thresholds.to_csv(
            os.path.join(
                output_path,
                f"elite_thresholds_{dataset_name.lower()}.csv",
            ),
            index=False,
        )
        all_retention.append(retention)
        all_thresholds.append(thresholds)
        survival_files.append(survival_file)

    combined_retention = pd.concat(all_retention, ignore_index=True)
    combined_thresholds = pd.concat(all_thresholds, ignore_index=True)
    combined_retention.to_csv(
        os.path.join(output_path, "elite_retention_yearly_combined.csv"),
        index=False,
    )
    combined_thresholds.to_csv(
        os.path.join(output_path, "elite_thresholds_combined.csv"),
        index=False,
    )

    period_statistics = calculate_period_statistics(
        combined_retention,
        bootstrap_iterations,
        bootstrap_block_length,
        random_seed,
    )
    period_statistics.to_csv(
        os.path.join(output_path, "elite_retention_period_statistics.csv"),
        index=False,
    )

    segmented_slopes = calculate_segmented_slopes(
        combined_retention,
        segmented_breakpoint_year,
    )
    segmented_slopes.to_csv(
        os.path.join(output_path, "elite_retention_segmented_slopes.csv"),
        index=False,
    )
 
    kaplan_meier_results = []
    for survival_file in survival_files:
        kaplan_meier_results.append(calculate_kaplan_meier(survival_file))
        
    combined_kaplan_meier = pd.concat(kaplan_meier_results, ignore_index=True)
    combined_kaplan_meier.to_csv(
        os.path.join(output_path, "elite_survival_kaplan_meier.csv"),
        index=False,
    )
    
    print(f"All results saved in: {output_path}")
