# Digital scholarly discovery stratified scientific attention

Analysis code for the manuscript *"Digital scholarly discovery stratified scientific attention"* by Leyang Xue, Shlomo Havlin,
Louis Shekhtman, and Alon Sela.

The code reproduces the citation-inequality, persistence, cross-field, and
reference-selection analyses reported in the paper, using bibliometric records
from the Microsoft Academic Graph (MAG) and OpenAlex.

> **Note.** This repository is being released incrementally. Some analysis and
> data-fetching scripts are not included; the raw MAG/OpenAlex corpora are not
> redistributed here (see **Data** below).

## Repository structure

Each analysis lives in its own folder. Within every folder, `code/` contains
the scripts and (where included) `figure/` and `result/` hold the rendered
figures and intermediate outputs.

```
.
├── figure1/     # citation inequality (Gini) across strata, 5-year window
├── figure2/     # persistence and survival of the top 1%
├── figure3/     # reference selection: prior visibility & semantic similarity
├── figure4/     # cross-field vs. same-field citation inequality
├── figureS1/    # Gini with a 10-year window (robustness)
├── figureS2/    # rank mobility within the citation summit
├── LICENSE
└── README.md
```

## What each folder computes

| Folder | Key scripts | Analysis | Manuscript figure* |
|--------|-------------|----------|--------------------|
| `figure1` | `calculate_c5_gini.py`, `gini_coefficient.py`, `plot_c5_gini.py` | Gini coefficient within the top 1%, 1–5%, 5–10%, 10–15% strata (5-year citation window) | Fig. 1 |
| `figure2` | `calculate_elite_persistence.py`, `plot_elite_persistence.py` | Top-1% forward retention and Kaplan–Meier survival | Fig. 3 |
| `figure3` | `calculate_figure3_*_metrics.py`, `plot_figure3_*.py` | Reference-selection analysis: prior top-1% visibility, semantic similarity, and top-cited fraction/counts | Fig. 4 (+ Supp.),Supplementary Fig. S3|
| `figure4` | `assign_field.py`, `build_topic_field.py`, `fig4_server_metrics.py`, `plot_fig4.py` | Same-field vs. cross-field citation inequality | Fig. 2 |
| `figureS1` | `calculate_c10_gini.py`, `plot_c10_gini.py` | Gini with a 10-year citation window | Supplementary Fig. S1 |
| `figureS2` | `calculate_rank_mobility.py`, `plot_rank_mobility.py` | Rank-mobility of top-1% papers | Supplementary Fig. S2 |

\* Repository folder numbers do not always match the final manuscript figure
numbers; the last column gives the correspondence to the published figures.

## Workflow

Within each folder the scripts follow the same two-step pattern:

1. `calculate_*.py` — read the (MAG/OpenAlex-derived) inputs and write the
   intermediate metrics (e.g. per-cohort Gini, retention, survival, or
   reference-selection tables).
2. `plot_*.py` — read those metrics and render the corresponding figure.

Run the `calculate_*` script first, then the matching `plot_*` script.

## Data

The analyses use two public bibliographic sources:

- **OpenAlex** — official bulk-data snapshot (downloaded 1 February 2023),
  https://openalex.org
- **Microsoft Academic Graph (MAG)** — an archived release of the database
  following the discontinuation of Microsoft Academic.

The full corpora are governed by their original terms and are **not**
redistributed in this repository. Paths to local data files in the scripts must
be adjusted to your environment.

## Requirements

The code is written in Python 3. Core dependencies include `numpy`, `pandas`,
`scipy`, and `matplotlib`; the semantic analyses additionally use
`transformers` and `torch` (SciBERT embeddings). See `requirements.txt` for the
full, version-pinned list.

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the paper:

> Xue, L., Havlin, S., Shekhtman, L. & Sela, A. Digital scholarly discovery stratified scientific attention.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
