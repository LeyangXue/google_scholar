#!/usr/bin/env python3
"""SI (FigureS4): robustness of the citation-semantics trend to the distance
measure (supports Figure 3b).

Three independent measures of citing--cited semantic proximity were computed on
the SAME sampled citation pairs and stored in parallel folders:
    year_topic_distance_masscenter  -> embedding mass-center DISTANCE  (higher = more distant)
    year_topic_distance_jaccard     -> concept-set Jaccard SIMILARITY   (higher = more similar)
    year_topic_distance_embeding    -> cosine SIMILARITY                (higher = more similar)

For each method and each citing year, we compute the mean over the 30 sample
files (each sample's mean, then the across-sample mean), and a moving-block
bootstrap 95% CI over the yearly series is used for the three historical-period
means. Direction differs by construction: a decline in mass-center distance is
equivalent to a rise in Jaccard/cosine similarity.

Outputs (figureS4/result/):
    semantic_methods_year.csv     per-year, per-method mean (+ across-sample CI)
    semantic_methods_period.csv   per-period mean + bootstrap 95% CI
"""

import glob
import math
import os
import re

import numpy as np
import pandas as pd

METHODS = {
    "masscenter": "Mass-center distance",
    "jaccard": "Jaccard similarity",
    "embeding": "Cosine similarity",
}
DIRECTION = {  # is a larger value MORE distant?
    "masscenter": "distance",
    "jaccard": "similarity",
    "embeding": "similarity",
}
BOOTSTRAP_ITER = 2000
BOOTSTRAP_BLOCK = 5
RANDOM_SEED = 20260712
MAX_YEAR = 2020   # analysis ends in 2020; later citing years are excluded


def period_name(year):
    if year <= 1979:
        return "Print-dominant era (1950–1979)"
    if year <= 2003:
        return "Digitization and networked-publishing transition (1980–2003)"
    return "Google Scholar era (2004 onward)"


def moving_block_bootstrap_ci(values, rng):
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


def method_year_means(src_root, method):
    """Return {year: (mean, ci_low, ci_high)} across the 30 sample files."""
    folder = os.path.join(src_root, f"year_topic_distance_{method}")
    files = glob.glob(os.path.join(folder, "topic_distance_*_k*.csv"))
    years = sorted({int(re.search(r"topic_distance_(\d+)_k", os.path.basename(f)).group(1))
                    for f in files})
    years = [y for y in years if y <= MAX_YEAR]
    out = {}
    for y in years:
        kfiles = sorted(glob.glob(os.path.join(folder, f"topic_distance_{y}_k*.csv")))
        sample_means = []
        for f in kfiles:
            v = pd.read_csv(f, usecols=["TopicDistance"])["TopicDistance"].astype(float)
            v = v[np.isfinite(v)]
            if len(v):
                sample_means.append(v.mean())
        if not sample_means:
            continue
        sm = np.array(sample_means)
        # across-sample 95% CI (percentile of the 30 sample means)
        lo, hi = np.quantile(sm, 0.025), np.quantile(sm, 0.975)
        out[y] = (sm.mean(), lo, hi, len(sm))
        print(f"[{method}] {y}: mean={sm.mean():.4f} (n_samples={len(sm)})", flush=True)
    return out


def main():
    src_root = "/Volumes/lydisk/work/work11/google_scholar/figure3/result/openalex"
    out_dir = "/Volumes/lydisk/work/work11/google_scholar/figureS4/result"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    year_rows, period_rows = [], []
    for method in METHODS:
        ym = method_year_means(src_root, method)
        for y, (mean, lo, hi, ns) in ym.items():
            year_rows.append({
                "method": method, "metric": METHODS[method],
                "direction": DIRECTION[method], "year": y,
                "mean": mean, "ci_lower": lo, "ci_upper": hi,
                "n_samples": ns, "period": period_name(y),
            })
        # period means + bootstrap CI over yearly means
        yr = pd.DataFrame([(y, v[0]) for y, v in ym.items()], columns=["year", "mean"])
        yr["period"] = yr["year"].map(period_name)
        for period, g in yr.groupby("period"):
            g = g.sort_values("year")
            lo, hi = moving_block_bootstrap_ci(g["mean"].to_numpy(), rng)
            period_rows.append({
                "method": method, "metric": METHODS[method],
                "direction": DIRECTION[method], "period": period,
                "start_year": int(g["year"].min()), "end_year": int(g["year"].max()),
                "n_years": len(g), "mean": g["mean"].mean(),
                "ci_lower": lo, "ci_upper": hi,
            })

    pd.DataFrame(year_rows).to_csv(os.path.join(out_dir, "semantic_methods_year.csv"), index=False)
    pd.DataFrame(period_rows).to_csv(os.path.join(out_dir, "semantic_methods_period.csv"), index=False)
    print("Saved FigureS4 result CSVs to", out_dir)


if __name__ == "__main__":
    main()
