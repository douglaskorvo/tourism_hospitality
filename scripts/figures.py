# -*- coding: utf-8 -*-
"""Manuscript figures."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

sys.path.insert(0, "/home/claude/hosp/scripts")
OUT = "/home/claude/hosp/out"
FIG = "/home/claude/hosp/fig"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a84"
GRID = "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.0,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def strip(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, lw=0.7)
        ax.yaxis.grid(False)


# ---------------------------------------------------------------- figure 1
def fig1():
    cs = pd.read_csv(f"{OUT}/city_scores.csv")
    var = pd.read_csv(f"{OUT}/city_index_variants.csv", index_col=0)
    cs = cs.set_index("city")
    cs["net"] = var["net_affect_rating"]
    cs = cs.sort_values("H_any_raw")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 6.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(cs))

    ax = axes[0]
    ax.hlines(y, cs.H_any_lo, cs.H_any_hi, color=BLUE, lw=2, alpha=.45)
    ax.plot(cs.H_any_raw, y, "o", ms=5.5, color=BLUE,
            markeredgecolor="white", markeredgewidth=.8, ls="none")
    ax.axvline(cs.H_any_raw.mean(), color=MUTED,
               lw=.9, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}" for c in cs.index], fontsize=8)
    ax.set_xlim(0.15, 0.60)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("reviews carrying a hospitality marker")
    ax.set_title("(a)  Observed prevalence", loc="left", fontsize=9,
                 color=INK, pad=8)
    strip(ax)
    for i, (lo, hi) in enumerate(zip(cs.H_any_lo, cs.H_any_hi)):
        pass
    ax.text(cs.H_any_raw.mean() + .004, len(cs) - .4,
            "corpus mean", fontsize=7, color=MUTED, va="center")

    ax = axes[1]
    order = cs.net
    ax.hlines(y, 0, order.values, color=MUTED, lw=1.0, alpha=.5)
    cols = [BLUE if v >= 0 else ORANGE for v in order.values]
    ax.scatter(order.values, y, s=34, c=cols, edgecolor="white", linewidth=.8,
               zorder=3)
    ax.axvline(0, color=INK2, lw=.9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("index net of affect and star rating (logit)")
    ax.set_title("(b)  Adjusted destination hospitality index", loc="left",
                 fontsize=9, color=INK, pad=8)
    strip(ax)

    fig.suptitle("Destination hospitality across 29 emerging-economy cities",
                 x=.02, ha="left", fontsize=10.5, y=1.005)
    fig.text(.02, -.015,
             "Left: share of reviews containing at least one induced "
             "hospitality marker; "
             "bars are 95% bootstrap intervals\nresampling suppliers (500 "
             "Right: city fixed effect from a logistic model that "
             "additionally removes "
             "generic\nexperiential affect and the star rating, centred on the corpus mean. "  # noqa: E501
             "n = 50,180 reviews, 11,477 suppliers.",
             fontsize=6.8, color=MUTED, va="top")
    fig.savefig(f"{FIG}/fig1_index.png")
    plt.close(fig)
    print("fig1 written")


# ---------------------------------------------------------------- figure 2
def fig2():
    prof = pd.read_csv(f"{OUT}/city_profiles.csv", index_col=0)
    prof = prof.sort_values("H_development")
    labels = ["Interpersonal conduct", "Created ambience", "Development"]
    cols = [BLUE, AQUA, ORANGE]
    keys = ["H_conduct", "H_ambience", "H_development"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    y = np.arange(len(prof))
    left = np.zeros(len(prof))
    gap = 0.002
    for k, c, lb in zip(keys, cols, labels):
        w = prof[k].values
        ax.barh(y, w - gap, left=left, height=.62, color=c, label=lb,
                edgecolor="white", linewidth=0)
        left = left + w
    for i in range(len(prof)):
        run = 0
        for k, c in zip(keys, cols):
            w = prof[k].values[i]
            ax.text(run + w / 2, i, f"{w*100:.0f}", ha="center", va="center",
                    fontsize=6.4, color="white", fontweight="bold")
            run += w
    ax.set_yticks(y)
    ax.set_yticklabels(prof.index, fontsize=8)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("composition of hospitality mentions within the city")
    strip(ax)
    ax.xaxis.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.06), ncol=3,
              frameon=False, fontsize=8, handlelength=1.2, handleheight=.9)
    ax.set_title("What kind of hospitality each destination is praised for",
                 loc="left", fontsize=10.5, color=INK, pad=26)
    fig.text(.02, -.02,
             "Share of each induced type among the hospitality-bearing "
             "reviews of the city, so the "
             "composition is independent\nof how expressive the city's review culture is. "  # noqa: E501
             "Values are percentages; rows sum to 100.",
             fontsize=6.8, color=MUTED, va="top")
    fig.savefig(f"{FIG}/fig2_profiles.png")
    plt.close(fig)
    print("fig2 written")


# ---------------------------------------------------------------- figure 3
def fig3():
    import warnings
    warnings.filterwarnings("ignore")
    from analysis import load
    from fastfe import fe_ols
    d = load().dropna(subset=["years_ago"])
    X = ["SQpos_flag", "SQneg_flag", "H_conduct", "H_ambience",
         "H_development", "EXP_flag"]
    tab, info = fe_ols(d, "DQpos_flag", X + ["log_len", "translated",
                                             "is_creative", "years_ago"],
                       "place_id")
    tab = tab.loc[X]
    names = ["Service quality (positive)", "Service quality (negative)",
             "Hospitality: interpersonal conduct",
             "Hospitality: created ambience",
             "Hospitality: development",
             "Generic experiential affect"]
    tab["lo"] = tab.coef - 1.96 * tab.se
    tab["hi"] = tab.coef + 1.96 * tab.se
    tab = tab.iloc[::-1]
    names = names[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(tab))
    sig = tab.p < .05
    cols = [BLUE if s else MUTED for s in sig]
    ax.hlines(y, tab.lo, tab.hi, color=cols, lw=2.2, alpha=.55)
    ax.scatter(tab.coef, y, s=46, c=cols, edgecolor="white", linewidth=.9,
               zorder=3)
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.2)
    ax.set_xlabel(
        "change in the probability that a review evaluates the destination")
    strip(ax)
    # value labels always sit to the right of the interval, so they can never
    # collide with the category labels on the left
    for i, r in enumerate(tab.itertuples()):
        txt = f"{r.coef:+.3f}" + ("  n.s." if r.p >= .05 else "")
        ax.text(r.hi + .0025, i, txt, fontsize=7.2,
                color=INK2 if r.p < .05 else MUTED, va="center", ha="left")
    ax.set_xlim(-.065, .088)
    ax.set_title("Hospitality carries the destination into the review; "
                 "service quality does not",
                 loc="left", fontsize=10.5, color=INK, pad=8)
    fig.text(.02, -.06,
             f"Linear probability model with supplier fixed effects, standard "
             f"errors clustered on the supplier. "
             f"n = {info['N']:,} reviews\nin {info['groups']:,} suppliers. "
             "Controls: review length, translation status, segment, recency. "
             "Bars are 95% intervals.",
             fontsize=6.8, color=MUTED, va="top")
    fig.savefig(f"{FIG}/fig3_rq2.png")
    plt.close(fig)
    print("fig3 written")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
