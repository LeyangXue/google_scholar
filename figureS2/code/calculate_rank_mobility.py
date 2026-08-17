#!/usr/bin/env python3
"""SI-2: rank mobility of the citation elite (supports Figure 2).

Two complementary measures of how frozen citation ranks became, computed per
publication-year cohort for MAG and OpenAlex, using the same cumulative-citation
elite definition as figure2 (A(a) = sum(C0..Ca)).

(1) Top-1% rank stability (Spearman):
    Among papers that are in the top 1% by cumulative citations at the baseline
    age, the Spearman correlation between their baseline-age rank and their
    follow-up-age rank. Two windows, matching figure2 retention:
        - 2-year: baseline age 2 vs follow-up age 4
        - 5-year: baseline age 5 vs follow-up age 10
    A rising correlation means the internal ordering of the elite became more
    frozen. Yearly per-cohort values + moving-block bootstrap CIs.

(2) Equal-width transition stay-probabilities:
    Within the top 5%, five equal 1%-wide bins (0-1, 1-2, 2-3, 3-4, 4-5) defined
    by rank at the baseline age. For each bin, the probability that a paper is in
    the SAME 1% bin at the follow-up age. Equal widths make bins directly
    comparable without normalization. Reported per cohort and pooled by period.

Outputs (figureS2/result/):
    spearman_topk_yearly.csv         per-cohort Spearman (both windows, both DBs)
    spearman_topk_period.csv         period means + bootstrap CIs
    transition_stay_yearly.csv       per-cohort stay-prob per 1% bin (both windows)
    transition_stay_period.csv       period-pooled stay-prob per 1% bin (+within-top5%)
"""

import glob
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---- configuration ----
TOP_FRACTION = 0.01                      # elite = top 1% (matches main-text usage)
WINDOWS = {"2yr": (2, 4), "5yr": (5, 10)}  # (baseline age, follow-up age)
# Largest publication cohort kept per window: the follow-up age must fall inside
# each database's coverage, otherwise a truncated window mechanically inflates
# the correlation. 5-year (age 10) is therefore capped earlier than 2-year.
MAX_COHORT = {"2yr": 2013, "5yr": 2010}
BIN_EDGES = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 1.00]  # 5 equal 1% bins + outside
BIN_LABELS = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%"]
BOOTSTRAP_ITER = 2000
BOOTSTRAP_BLOCK = 5
RANDOM_SEED = 20260712


def period_name(year):
    """Historical periods, matching Figure 1/Figure 2 boundaries exactly."""
    if year <= 1979:
        return "Print-dominant era (1950–1979)"
    if year <= 2003:
        return "Digitization and networked-publishing transition (1980–2003)"
    return "Google Scholar era (2004 onward)"


def cumulative_at(citations, age):
    """Cumulative citations A(age) = sum of C0..C_age (age inclusive)."""
    return citations[:, : age + 1].sum(axis=1)


def rank_percentile(values):
    """Return each paper's percentile position (0 = most cited)."""
    order = np.argsort(-values, kind="stable")
    perc = np.empty(len(values), dtype=np.float64)
    perc[order] = np.arange(len(values)) / len(values)
    return perc


def load_citation_matrix(input_file, max_needed_age):
    """Load PaperID + C0..C_max as an int matrix; return (ids, citations)."""
    header = pd.read_csv(input_file, nrows=0).columns.tolist()
    age_cols = []
    for age in range(max_needed_age + 1):
        col = f"C{age}"
        if col not in header:
            raise ValueError(f"{input_file} is missing {col}")
        age_cols.append(col)
    data = pd.read_csv(input_file, usecols=age_cols, dtype=np.int32)
    return data[age_cols].to_numpy(dtype=np.int64, copy=False)


def cohort_spearman(citations, baseline_age, followup_age, top_fraction):
    """Spearman between baseline-age and follow-up-age cumulative citations,
    restricted to papers in the top `top_fraction` at the baseline age."""
    a_base = cumulative_at(citations, baseline_age)
    a_follow = cumulative_at(citations, followup_age)
    n = len(a_base)
    k = max(2, int(math.ceil(n * top_fraction)))
    elite_idx = np.argpartition(a_base, n - k)[n - k:]
    if elite_idx.size < 3:
        return np.nan, int(elite_idx.size)
    rho = spearmanr(a_base[elite_idx], a_follow[elite_idx]).correlation
    return rho, int(elite_idx.size)


def cohort_transition(citations, baseline_age, followup_age):
    """Stay-in-same-1%-bin probability for each of the five equal top-5% bins,
    plus the probability that top 0-1% papers remain anywhere within top 5%."""
    s_base = np.digitize(rank_percentile(cumulative_at(citations, baseline_age)),
                         BIN_EDGES[1:-1])
    s_follow = np.digitize(rank_percentile(cumulative_at(citations, followup_age)),
                           BIN_EDGES[1:-1])
    n_states = len(BIN_EDGES) - 1  # 6 (5 bins + outside)
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    for i in range(n_states):
        mask = s_base == i
        if mask.any():
            counts[i] = np.bincount(s_follow[mask], minlength=n_states)
    return counts


def moving_block_bootstrap_ci(values, rng):
    """95% CI of the mean via moving-block bootstrap (matches figure2)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return values[0], values[0]
    block = min(BOOTSTRAP_BLOCK, n)
    n_blocks = int(math.ceil(n / block))
    max_start = n - block
    est = np.empty(BOOTSTRAP_ITER)
    for it in range(BOOTSTRAP_ITER):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[s:s + block] for s in starts])[:n]
        est[it] = sample.mean()
    return np.quantile(est, 0.025), np.quantile(est, 0.975)


def process_dataset(data_path, dataset_name, start_year, end_year):
    files = sorted(glob.glob(os.path.join(data_path, "*_citation.csv")))
    spearman_rows, transition_counts = [], {}  # transition_counts[(win)][year] = 6x6
    for f in files:
        year = int(os.path.basename(f).split("_")[0])
        if not (start_year <= year <= end_year):
            continue
        max_age_needed = max(fu for _, fu in WINDOWS.values())
        try:
            citations = load_citation_matrix(f, max_age_needed)
        except ValueError:
            continue  # cohort too young for the 10-age window
        if citations.shape[1] <= max_age_needed:
            continue
        for win, (ba, fu) in WINDOWS.items():
            if year > MAX_COHORT[win]:
                continue
            rho, k = cohort_spearman(citations, ba, fu, TOP_FRACTION)
            spearman_rows.append({
                "Dataset": dataset_name, "Publication_year": year,
                "Period": period_name(year), "Window": win,
                "Baseline_age": ba, "Followup_age": fu,
                "Elite_n": k, "Spearman": rho,
            })
            counts = cohort_transition(citations, ba, fu)
            transition_counts.setdefault(win, {})[year] = counts
        del citations
        print(f"[{dataset_name}] {year} done", flush=True)
    return pd.DataFrame(spearman_rows), transition_counts


def transition_yearly_rows(transition_counts, dataset_name):
    rows = []
    for win, per_year in transition_counts.items():
        for year, counts in per_year.items():
            for i, lab in enumerate(BIN_LABELS):
                total = counts[i].sum()
                if total == 0:
                    continue
                rows.append({
                    "Dataset": dataset_name, "Publication_year": year,
                    "Period": period_name(year), "Window": win,
                    "Bin": lab, "N_baseline": int(total),
                    "Stay_prob": counts[i, i] / total,
                    "Within_top5_prob": counts[i, :5].sum() / total,
                })
    return pd.DataFrame(rows)


def transition_period_rows(transition_counts, dataset_name, rng):
    """Period-level stay-probabilities as the UNWEIGHTED mean of the per-cohort
    stay-probabilities (each publication year counts equally), matching how the
    Spearman period means and Figure 2 period means are computed. A moving-block
    bootstrap over the yearly series gives the 95% CI."""
    rows = []
    for win, per_year in transition_counts.items():
        # per_year[year] is a 6x6 count matrix; convert to per-year stay-prob.
        for i, lab in enumerate(BIN_LABELS):
            by_period = {}
            for year, counts in per_year.items():
                total = counts[i].sum()
                if total == 0:
                    continue
                by_period.setdefault(period_name(year), []).append(
                    (year, counts[i, i] / total, counts[i, :5].sum() / total))
            for period, triples in by_period.items():
                triples.sort()
                stay = np.array([t[1] for t in triples])
                within = np.array([t[2] for t in triples])
                lo, hi = moving_block_bootstrap_ci(stay, rng)
                rows.append({
                    "Dataset": dataset_name, "Period": period, "Window": win,
                    "Bin": lab, "N_years": len(stay),
                    "Stay_prob": stay.mean(),
                    "Stay_CI_lower": lo, "Stay_CI_upper": hi,
                    "Within_top5_prob": within.mean(),
                })
    return pd.DataFrame(rows)


def spearman_period_rows(spearman_df, rng):
    rows = []
    for (dataset, window, period), g in spearman_df.groupby(
        ["Dataset", "Window", "Period"], sort=False
    ):
        vals = g["Spearman"].to_numpy(dtype=float)
        lo, hi = moving_block_bootstrap_ci(vals, rng)
        rows.append({
            "Dataset": dataset, "Window": window, "Period": period,
            "Start_year": int(g["Publication_year"].min()),
            "End_year": int(g["Publication_year"].max()),
            "N_years": len(g), "Mean_spearman": np.nanmean(vals),
            "CI_lower": lo, "CI_upper": hi,
        })
    return pd.DataFrame(rows)


def main():
    mag_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/result/mag/year_citation_dynamic"
    oa_path = "/Volumes/lydisk/work/work11/google_scholar/figure2/result/openalex/year_citation_dynamic"
    out = "/Volumes/lydisk/work/work11/google_scholar/figureS2/result"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    all_spearman, all_trans_yearly, all_trans_period = [], [], []
    for name, path in [("MAG", mag_path), ("OpenAlex", oa_path)]:
        sp, tc = process_dataset(path, name, 1950, 2023)
        all_spearman.append(sp)
        all_trans_yearly.append(transition_yearly_rows(tc, name))
        all_trans_period.append(transition_period_rows(tc, name, rng))

    spearman = pd.concat(all_spearman, ignore_index=True)
    spearman.to_csv(os.path.join(out, "spearman_topk_yearly.csv"), index=False)
    spearman_period_rows(spearman, rng).to_csv(
        os.path.join(out, "spearman_topk_period.csv"), index=False)
    pd.concat(all_trans_yearly, ignore_index=True).to_csv(
        os.path.join(out, "transition_stay_yearly.csv"), index=False)
    pd.concat(all_trans_period, ignore_index=True).to_csv(
        os.path.join(out, "transition_stay_period.csv"), index=False)
    print("Saved SI-2 result CSVs to", out)


if __name__ == "__main__":
    main()
