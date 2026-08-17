"""
build_topic_field.py
---------------------
Attach FIELD labels to every citation pair in the Fig3 topic-distance samples,
merge all k-samples per year, and write per-year Parquet files + a summary CSV.

Pipeline (per citing year Y)
============================
  1. Build citing field map from   year_citingpaper_concept/{Y}_..._concept[_sample].pkl
  2. Build cited  field map from   year_citedpaper_concept/{Y}_..._concept[_sample].pkl
       (the cited pickle is CUMULATIVE: it holds every cited paper published <= Y,
        so a single file per citing year covers ~100% of that year's cited IDs.)
  3. For each  year_topic_distance_embeding/topic_distance_{Y}_k{K}.csv  (K = 0..29):
       - map CitingPublicationId -> FieldCiting
       - map CitedPublicationId  -> FieldCited
       - tag SampleK = K
  4. Concatenate the 30 samples, add CrossField = (FieldCiting != FieldCited),
     and save -> year_topic_field/topic_field_{Y}.parquet
  5. Accumulate a summary row (match rate, cross-field share, mean topic distance).

Concept-file selection
======================
Prefer the *_sample.pkl (smaller, and it matches the sampled papers) but verify
coverage against the IDs actually needed; if coverage is poor, fall back to the
full (non-sample) file. This handles years like 1985 whose *sample* citing file
is a mismatched subsample (~6% coverage) while its full file covers 100%.

Output columns
==============
CitingPublicationId, CitingYear, CitedPublicationId, CitedYear,
TopicDistance, SampleK, FieldCiting, FieldCited, CrossField

Usage
=====
  python build_topic_field.py
Adjust SRC / OUT below if paths differ.
"""

import os
import re
import gc
import glob
import pandas as pd

from assign_field import build_field_map

# --- paths (edit if needed) ---
SRC = "../../figure3/result/openalex"          # dir holding the concept + topic_distance folders
OUT = "../result/openalex"                      # dir to write field-matched outputs
TD_DIR = os.path.join(SRC, "year_topic_distance_embeding")

COVERAGE_MIN = 0.50   # if a concept file covers < this fraction of needed IDs, try the other variant


def _pid(x):
    return str(int(x))


def concept_field_map(side, year, needed_ids):
    """Return {paper_id: field} for `side` in {'citing','cited'} and `year`,
    choosing whichever concept file (sample first, then full) best covers
    `needed_ids`."""
    candidates = [
        f"{SRC}/year_{side}paper_concept/{year}_{side}paper_concept_sample.pkl",
        f"{SRC}/year_{side}paper_concept/{year}_{side}paper_concept.pkl",
    ]
    best_map, best_cov = {}, -1.0
    for path in candidates:
        if not os.path.exists(path):
            continue
        fm = build_field_map(pd.read_pickle(path))
        cov = (len(needed_ids & set(fm)) / len(needed_ids)) if needed_ids else 0.0
        if cov > best_cov:
            best_map, best_cov = fm, cov
        if cov >= 0.999:            # good enough, stop early
            break
        gc.collect()
    return best_map, best_cov


def process_year(year):
    
    kfiles = sorted(
        glob.glob(f"{TD_DIR}/topic_distance_{year}_k*.csv"),
        key=lambda f: int(re.search(r"_k(\d+)\.csv", f).group(1)),
    )
    if not kfiles:
        return None

    # Collect the IDs we actually need (to pick the right concept file).
    need_citing, need_cited, raw = set(), set(), []
    for f in kfiles:
        t = pd.read_csv(f)
        t["SampleK"] = int(re.search(r"_k(\d+)\.csv", f).group(1))
        need_citing |= set(t["CitingPublicationId"].apply(_pid))
        need_cited |= set(t["CitedPublicationId"].apply(_pid))
        raw.append(t)

    fc, cov_c = concept_field_map("citing", year, need_citing)
    fd, cov_d = concept_field_map("cited", year, need_cited)

    df = pd.concat(raw, ignore_index=True)
    df["FieldCiting"] = df["CitingPublicationId"].apply(_pid).map(fc)
    df["FieldCited"] = df["CitedPublicationId"].apply(_pid).map(fd)

    # CrossField: True/False when both fields present, else <NA>.
    both = df["FieldCiting"].notna() & df["FieldCited"].notna()
    df["CrossField"] = pd.NA
    df.loc[both, "CrossField"] = (df.loc[both, "FieldCiting"] != df.loc[both, "FieldCited"])

    os.makedirs(f"{OUT}/year_topic_field", exist_ok=True)
    df.to_parquet(f"{OUT}/year_topic_field/topic_field_{year}.parquet", index=False)

    m = df[both]
    row = dict(
        year=year,
        n_pairs=len(df),
        n_samples=len(kfiles),
        citing_coverage=round(cov_c, 4),
        cited_coverage=round(cov_d, 4),
        match_rate=round(len(m) / len(df), 4),
        cross_field_share=round((m["FieldCiting"] != m["FieldCited"]).mean(), 4),
        mean_topic_distance=round(df["TopicDistance"].astype(float).mean(), 4),
    )
    del raw, df, m, fc, fd
    gc.collect()
    return row


def main():

    years = sorted({
        int(re.match(r"topic_distance_(\d+)_k", os.path.basename(f)).group(1))
        for f in glob.glob(f"{TD_DIR}/topic_distance_*_k*.csv")
    })
    summary = []
    for y in years:
        row = process_year(y)
        if row:
            summary.append(row)
            print(f"{y}: pairs={row['n_pairs']:>8}  match={row['match_rate']*100:5.1f}%  "
                  f"cross_field={row['cross_field_share']:.3f}")
    os.makedirs(f"{OUT}/summary", exist_ok=True)
    pd.DataFrame(summary).to_csv(f"{OUT}/summary/cross_field_by_year.csv", index=False)
    print(f"\nWrote {len(summary)} years -> {OUT}/year_topic_field/ and summary/cross_field_by_year.csv")


if __name__ == "__main__":
    main()
