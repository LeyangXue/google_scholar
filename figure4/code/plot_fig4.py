#!/usr/bin/env python3
"""Figure 4 (1x3): cross- vs same-field concentration (Gini) of citations.

(a) Top 1%: Gini of cross-field (red) vs same-field (blue) citations over time.
(b) Same-field citations: Gini by citation stratum (top1%, 1-5%, 5-10%, 10-15%).
(c) Cross-field citations: Gini by citation stratum.
Every line carries three-period means; strata use the Figure-1 palette.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULT = "/Volumes/lydisk/work/work11/google_scholar/figure4/result/openalex/result"
OUTDIR = "/Volumes/lydisk/work/work11/google_scholar/figure4/figure"

CROSS_C, SAME_C = "#E66101", "#5E3C99"   # orange / purple (distinct from strata palette)
STRATA = ["top1%", "1-5%", "5-10%", "10-15%"]
STRAT_COLOR = {"top1%": "#3C5488", "1-5%": "#E64B35",
               "5-10%": "#00A087", "10-15%": "#C49A32"}
STRAT_LABEL = {"top1%": "Top 1%", "1-5%": "Top 1–5%",
               "5-10%": "Top 5–10%", "10-15%": "Top 10–15%"}
CMAX = 2018
BOOT_ITER, BOOT_BLOCK, SEED = 2000, 5, 20260812
PERIODS = ["Pre", "Dig", "GS"]
PERIOD_BOUNDS = {"Pre": (1950, 1979), "Dig": (1980, 2003), "GS": (2004, CMAX)}


SMOOTH_W = 3  # centred rolling-mean window for the plotted lines (means use raw)


def per(y):
    return "Pre" if y <= 1979 else ("Dig" if y <= 2003 else "GS")


def smooth(y):
    return pd.Series(y).rolling(SMOOTH_W, center=True, min_periods=1).mean().to_numpy()


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=7.5, length=3, width=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)


def add_eras(ax):
    ax.axvspan(1980, 2004, color="#D9DDE3", alpha=0.60, linewidth=0, zorder=0)
    ax.axvspan(2004, CMAX, color="#EFEFEF", alpha=0.78, linewidth=0, zorder=0)


def boot_ci(vals, rng):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return vals[0], vals[0]
    b = min(BOOT_BLOCK, n); nb = int(math.ceil(n / b)); est = np.empty(BOOT_ITER)
    for i in range(BOOT_ITER):
        s = rng.integers(0, n - b + 1, size=nb)
        est[i] = np.concatenate([vals[j:j + b] for j in s])[:n].mean()
    return np.quantile(est, 0.025), np.quantile(est, 0.975)


def draw_line_with_periods(ax, d, col, color, rng, val_fs, above=True, dy_extra=0.0):
    d = d.sort_values("cohort")
    ax.plot(d["cohort"], smooth(d[col].to_numpy()), color=color, linewidth=1.8,
            solid_capstyle="round", zorder=4)
    d2 = d.copy(); d2["P"] = d2["cohort"].map(per)
    span = np.nanmax(d[col].to_numpy()) - np.nanmin(d[col].to_numpy())
    off = 0.045 * max(span, 0.02)
    for p in PERIODS:
        g = d2[d2["P"] == p]
        if g.empty:
            continue
        s, e = PERIOD_BOUNDS[p]
        m = g[col].mean(); lo, hi = boot_ci(g[col].to_numpy(), rng)
        ax.fill_between([s, e], [lo, lo], [hi, hi], color=color, alpha=0.08,
                        linewidth=0, zorder=1)
        # faint, receding period-mean segment (behind the trend line)
        ax.hlines(m, s, e, color=color, linewidth=1.3, alpha=0.45, zorder=2)
        if val_fs:
            y, va = (m + off, "bottom") if above else (m - off, "top")
            y += dy_extra
            ax.text((s + e) / 2, y, f"{m:.2f}", color=color, fontsize=val_fs,
                    ha="center", va=va, zorder=6)


def top1_panel(ax, c, rng):
    d = c[c["stratum"] == "top1%"][["cohort", "gini_cross", "gini_same"]]
    add_eras(ax)
    draw_line_with_periods(ax, d, "gini_cross", CROSS_C, rng, 6.6, above=True)
    # purple (same-field) value labels shifted further DOWN in all three periods
    draw_line_with_periods(ax, d, "gini_same", SAME_C, rng, 6.6, above=False,
                           dy_extra=-0.011)
    finalize(ax, "Top 1%: cross vs same", "Gini of citations (top 1%)")
    ax.legend([Line2D([0], [0], color=CROSS_C, lw=2.2),
               Line2D([0], [0], color=SAME_C, lw=2.2)],
              ["Cross-field", "Same-field"], loc="upper left", frameon=False,
              fontsize=6.8, handlelength=1.3, borderaxespad=0.3)


def _period_mean(d, col, p):
    s, e = PERIOD_BOUNDS[p]
    g = d[(d["cohort"] >= s) & (d["cohort"] <= e)]
    return g[col].mean() if len(g) else np.nan


LOWER = ["1-5%", "5-10%", "10-15%"]

# Fine manual nudges for period-mean value labels, keyed (field, period, stratum).
#   xf : x-position as a fraction along the period segment (default 0.5 = centre)
#   ha : horizontal anchor (default "center")
#   dy : vertical shift in units of the panel's data span (+ up, - down)
LABEL_OFFSETS = {
    # (b) same-field
    ("same", "Pre", "1-5%"):  {"xf": 0.04, "ha": "left"},
    ("same", "Pre", "5-10%"): {"xf": 0.04, "ha": "left"},
    ("same", "Pre", "10-15%"): {"xf": 0.04, "ha": "left"},
    ("same", "Dig", "1-5%"):  {"xf": 0.82},
    ("same", "Dig", "10-15%"): {"dy": -0.09},
    ("same", "GS", "1-5%"):   {"dy": 0.07},
    ("same", "GS", "5-10%"):  {"dy": 0.05},
    ("same", "GS", "10-15%"): {"dy": -0.13},
    # (c) cross-field
    ("cross", "Dig", "5-10%"): {"dy": -0.06},
    ("cross", "Dig", "1-5%"):  {"dy": -0.06},
    ("cross", "GS", "1-5%"):   {"dy": 0.06},
    ("cross", "GS", "5-10%"):  {"dy": -0.07},
    ("cross", "GS", "10-15%"):  {"xf": 1.2, "ha": "right"},
}


def strata_panel(ax, c, field, rng, title, ylab):
    """Only the three non-top-1% strata (top 1% is shown in panel a). With top 1%
    removed the y-axis zooms to these three, spreading them apart. Every line's
    three period means are labelled, spread into the whitespace above/below."""
    
    add_eras(ax)
    col = f"gini_{field}"
    xfrac = {"1-5%": 0.20, "5-10%": 0.50, "10-15%": 0.80}
    means = {s: {} for s in LOWER}
    
    allvals = []
    for s in LOWER:
        d = c[c["stratum"] == s][["cohort", col]].sort_values("cohort")
        ys = smooth(d[col].to_numpy()); allvals.append(ys)
        ax.plot(d["cohort"], ys, color=STRAT_COLOR[s], linewidth=1.9,
                alpha=0.95, solid_capstyle="round", zorder=4)
        for p in PERIODS:
            s0, e0 = PERIOD_BOUNDS[p]
            seg = d[(d["cohort"] >= s0) & (d["cohort"] <= e0)][col].to_numpy()
            m = _period_mean(d, col, p); means[s][p] = m
            lo, hi = boot_ci(seg, rng)
            ax.fill_between([s0, e0], [lo, lo], [hi, hi], color=STRAT_COLOR[s],
                            alpha=0.07, linewidth=0, zorder=1)
            # faint, receding period-mean segment (behind the trend lines)
            ax.hlines(m, s0, e0, color=STRAT_COLOR[s], linewidth=1.3,
                      alpha=0.45, zorder=2)
            
    span = np.nanmax(np.concatenate(allvals)) - np.nanmin(np.concatenate(allvals))
    gap = 0.058 * span   # min vertical spacing so centred labels never overlap
    for p in PERIODS:
        s0, e0 = PERIOD_BOUNDS[p]; width = e0 - s0
        # base: place value at the segment centre, nudged only enough to declutter
        order = sorted(LOWER, key=lambda s: means[s][p])  # ascending
        ys, prev = {}, None
        for s in order:
            y = means[s][p] if prev is None else max(means[s][p], prev + gap)
            ys[s] = y; prev = y
        for s in LOWER:
            ov = LABEL_OFFSETS.get((field, p, s), {})
            x = s0 + ov.get("xf", 0.5) * width
            y = ys[s] + ov.get("dy", 0.0) * span
            ax.text(x, y, f"{means[s][p]:.3f}", color=STRAT_COLOR[s],
                    fontsize=6.2, ha=ov.get("ha", "center"), va="bottom",
                    zorder=9)
    finalize(ax, title, ylab)


def finalize(ax, title, ylab, xmax=CMAX):
    ax.set_xlim(1950, xmax); ax.set_xticks([1960, 1980, 2000, 2020])
    ymin, ymax = ax.get_ylim()
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title(title, fontsize=9.4, pad=5)
    ax.set_xlabel("Publication year", fontsize=8.8)
    ax.set_ylabel(ylab, fontsize=8.8)
    style_axis(ax)


def main():
    plt.rcParams.update({
        "font.family": "Arial", "axes.labelcolor": "#262626",
        "xtick.color": "#262626", "ytick.color": "#262626",
        "text.color": "#262626", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    c = pd.read_csv(os.path.join(RESULT, "fig4_c_gini.csv"))
    c = c[c["cohort"] <= CMAX]
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.5))
    top1_panel(ax[0], c, rng)
    strata_panel(ax[1], c, "same", rng, "Same-field citations",
                 "Gini of same-field citations")
    strata_panel(ax[2], c, "cross", rng, "Cross-field citations",
                 "Gini of cross-field citations")

    fig.subplots_adjust(left=0.068, right=0.988, top=0.905, bottom=0.255, wspace=0.30)
    for a, lab in zip(ax, ["(a)", "(b)", "(c)"]):
        pos = a.get_position()
        fig.text(pos.x0 - 0.045, pos.y1 + 0.02, lab, fontsize=11,
                 fontweight="bold", ha="left", va="bottom")

    strat_handles = [Line2D([0], [0], color=STRAT_COLOR[s], lw=2.2,
                            label=STRAT_LABEL[s]) for s in LOWER]
    era_handles = [Patch(facecolor="#FFFFFF", edgecolor="#BDBDBD", linewidth=0.6),
                   Patch(facecolor="#D9DDE3", alpha=0.90, edgecolor="none"),
                   Patch(facecolor="#EFEFEF", alpha=0.95, edgecolor="none")]
    era_labels = ["Print-dominant era (1950–1979)",
                  "Digitization and networked-publishing transition (1980–2003)",
                  "Google Scholar era (2004 onward)"]
    # row 1: strata; row 2: the three historical periods (all on one line)
    fig.legend(handles=strat_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.53, 0.085), columnspacing=1.6,
               handlelength=1.4)
    fig.legend(era_handles, era_labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7.0, bbox_to_anchor=(0.53, 0.020), columnspacing=1.3,
               handlelength=1.3)

    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "Figure4_crossfield.png"), dpi=600)
    fig.savefig(os.path.join(OUTDIR, "Figure4_crossfield.pdf"))
    print("saved figure to", OUTDIR)


if __name__ == "__main__":
    
    main()
