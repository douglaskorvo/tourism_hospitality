# -*- coding: utf-8 -*-
"""
RQ1  Can the hospitality of a city be assessed from visitor reviews?
RQ2  Is perceived service quality related to perceived destination quality?

Design
------
Reviews (level 1) are nested in suppliers (level 2) nested in cities
(level 3) nested in countries (level 4).  Every city-level quantity is
therefore reported (a) raw, (b) adjusted for corpus composition -- segment,
review length, translation status and recency -- and (c) with a bootstrap
confidence interval that resamples SUPPLIERS, not reviews, because reviews
of the same supplier are not independent.
"""
import sys
import warnings
import zlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "/home/claude/hosp/scripts")
from fastfe import fe_ols  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "/home/claude/hosp/out"
RNG = np.random.default_rng(2026)
NBOOT = 500


# ------------------------------------------------------------------ loading
def load():
    d = pd.read_parquet(f"{OUT}/reviews_coded.parquet")
    M = np.load(f"{OUT}/marker_matrix.npy")
    typ = pd.read_csv(f"{OUT}/typology.csv")
    mk = pd.read_csv(f"{OUT}/markers.csv")
    assert len(d) == len(M), (len(d), len(M))

    order = {f: i for i, f in enumerate(mk.family)}
    names = {0: "development", 1: "conduct", 2: "ambience"}
    for t, nm in names.items():
        cols = [order[f] for f in typ[typ.type == t].family if f in order]
        d[f"H_{nm}_n"] = M[:, cols].sum(axis=1)
        d[f"H_{nm}"] = (M[:, cols].sum(axis=1) > 0).astype(int)
    d["H_any"] = ((d.H_development + d.H_conduct +
                  d.H_ambience) > 0).astype(int)
    d["H_breadth"] = d.H_development + d.H_conduct + d.H_ambience

    d["log_len"] = np.log1p(d.n_tokens)
    d["years_ago"] = d.time_desc.map(parse_age)
    d["is_creative"] = (d.segment == "creative").astype(int)
    d["high_rating"] = (d.rating == 5).astype(int)
    return d


def parse_age(s):
    """Google serves relative dates; convert to approximate years."""
    s = str(s)
    num = 1.0
    m = pd.Series([s]).str.extract(r"(\d+)")[0].iloc[0]
    if pd.notna(m):
        num = float(m)
    if "ano" in s:
        return num
    if "mes" in s or "mês" in s or "meses" in s:
        return num / 12
    if "semana" in s:
        return num / 52
    if "dia" in s:
        return num / 365
    return np.nan


# ------------------------------------------------- city scores + bootstrap
def city_scores(d, col):
    """Raw prevalence and supplier-bootstrap CI for one indicator.

    The bootstrap resamples suppliers with replacement and recomputes the
    city mean as sum(supplier sums) / sum(supplier counts), which is the
    cluster bootstrap appropriate to reviews nested in suppliers."""
    rows = []
    for city, g in d.groupby("city", observed=True):
        agg = g.groupby("place_id", observed=True)[col].agg(["sum", "size"])
        sv, nv = agg["sum"].values.astype(
            float), agg["size"].values.astype(float)
        P = len(sv)
        idx = rng_for(col, "bootstrap" + city).integers(0, P, size=(NBOOT, P))
        boots = sv[idx].sum(axis=1) / nv[idx].sum(axis=1)
        places = agg.index.values
        rows.append({
            "city": city, "country": g.country.iloc[0],
            "n_reviews": len(g), "n_suppliers": len(places),
            f"{col}_raw": g[col].mean(),
            f"{col}_lo": np.percentile(boots, 2.5),
            f"{col}_hi": np.percentile(boots, 97.5),
            f"{col}_se": boots.std(ddof=1),
        })
    return pd.DataFrame(rows)


def adjusted_city_index(d, outcome):
    """City fixed effects net of corpus composition, on the logit scale."""
    dd = d.dropna(subset=["years_ago"]).copy()
    mod = smf.logit(f"{outcome} ~ C(city) + is_creative + log_len + "
                    f"translated + years_ago", data=dd).fit(disp=0)
    base = dd.city.cat.categories[0] if hasattr(dd.city, "cat") \
        else sorted(dd.city.unique())[0]
    eff = {base: 0.0}
    for name, val in mod.params.items():
        if name.startswith("C(city)[T."):
            eff[name.split("C(city)[T.")[1].rstrip("]")] = val
    s = pd.Series(eff, name=f"{outcome}_adj_logit")
    return s - s.mean(), mod


# ------------------------------------------------------------- reliability
def rng_for(col, salt=""):
    """A generator seeded from the indicator name, so a statistic does not
    depend on how many random draws happened earlier in the script."""
    return np.random.default_rng(zlib.crc32((col + salt).encode()))


def split_half(d, col, reps=200):
    """Spearman-Brown corrected split-half reliability of the city ranking.
    Splits are taken over SUPPLIERS so the halves are genuinely independent."""
    rng = rng_for(col, "split_half")
    agg = {}
    for city, g in d.groupby("city", observed=True):
        a = g.groupby("place_id", observed=True)[col].agg(["sum", "size"])
        agg[city] = (a["sum"].values.astype(float),
                     a["size"].values.astype(float))
    cities = sorted(agg)
    rs = []
    for _ in range(reps):
        h1, h2 = [], []
        for c in cities:
            sv, nv = agg[c]
            P = len(sv)
            perm = rng.permutation(P)
            i1, i2 = perm[:P // 2], perm[P // 2:]
            if nv[i1].sum() > 20 and nv[i2].sum() > 20:
                h1.append(sv[i1].sum() / nv[i1].sum())
                h2.append(sv[i2].sum() / nv[i2].sum())
        if len(h1) > 5:
            rs.append(stats.pearsonr(h1, h2)[0])
    # average the split-half correlations first, then apply Spearman-Brown
    # once -- correcting each replication and averaging afterwards is a
    # different (and non-standard) quantity, and would not reproduce from
    # the correlation reported in the supplementary workbook

    def sb(x):
        return 2 * x / (1 + x)
    return sb(float(np.mean(rs))), sb(float(np.percentile(rs, 2.5))), \
        sb(float(np.percentile(rs, 97.5)))


def between_city_test(d, col, nperm=1000):
    """Do cities differ by more than chance?  ICC(1) plus a permutation test
    that shuffles city labels across suppliers."""
    rng = rng_for(col, "permutation")
    g = d.groupby("city", observed=True)[col]
    k = g.size()
    m = g.mean()
    grand = d[col].mean()
    obs = float(np.average((m - grand) ** 2, weights=k))

    sup = d.groupby("place_id", observed=True).agg(
        city=("city", "first"), s=(col, "sum"), n=(col, "size"))
    codes, _ = pd.factorize(sup.city)
    C = codes.max() + 1
    sv, nv = sup.s.values.astype(float), sup.n.values.astype(float)

    null = np.empty(nperm)
    for i in range(nperm):
        pc = rng.permutation(codes)
        ms = np.bincount(pc, weights=sv, minlength=C)
        mn = np.bincount(pc, weights=nv, minlength=C)
        null[i] = float(np.average((ms / mn - grand) ** 2, weights=mn))
    p = (1 + (null >= obs).sum()) / (nperm + 1)

    # ICC(1) from a one-way random-effects decomposition over suppliers
    val = sv / nv
    dfm = pd.DataFrame({"c": codes, "val": val, "n": nv})
    gm = dfm.groupby("c").apply(
        lambda x: np.average(x.val, weights=x.n))
    sz = dfm.groupby("c").size()
    ms_b = float((sz * (gm - grand) ** 2).sum() / (C - 1))
    ms_w = float(dfm.groupby("c").val.var(ddof=1).mean())
    n0 = float(sz.mean())
    icc = (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)
    return obs, float(p), float(icc)


# --------------------------------------------------------------------- main
def main():
    d = load()
    print(f"analysis corpus: {len(d):,} reviews | "
          f"{d.place_id.nunique():,} suppliers | {d.city.nunique()} cities | "
          f"{d.country.nunique()} countries\n")

    print("=" * 72)
    print("RQ1  MEASURABILITY")
    print("=" * 72)
    prev = d[["H_any", "H_conduct", "H_ambience", "H_development",
              "DESTHOSP_flag", "SQpos_flag", "SQneg_flag",
              "DQpos_flag", "DQneg_flag", "EXP_flag"]].mean()
    print("\nreview-level prevalence")
    print(prev.round(4).to_string())
    print(f"\nmean number of hospitality types per review: "
          f"{d.H_breadth.mean():.3f}")

    print("\n--- reliability of the city ranking (Spearman-Brown) ---")
    rel = {}
    for c in ["H_any", "H_conduct", "H_ambience", "H_development",
              "SQpos_flag", "DQpos_flag"]:
        rel[c] = split_half(d, c)
        print(f"  {c:16s} r_sb = {rel[c][0]:.3f}  "
              f"[{rel[c][1]:.3f}, {rel[c][2]:.3f}]")

    print("\n--- do cities differ beyond chance? ---")
    disc = {}
    for c in ["H_any", "H_conduct", "H_ambience", "H_development",
              "SQpos_flag", "DQpos_flag"]:
        obs, p, icc = between_city_test(d, c)
        disc[c] = (obs, p, icc)
        print(f"  {c:16s} between-city var {obs:.5f}  p = {p:.4f}  "
              f"ICC(1) = {icc:.3f}")

    print("\n--- city hospitality index ---")
    cs = city_scores(d, "H_any")
    for c in ["H_conduct", "H_ambience", "H_development", "SQpos_flag",
              "SQneg_flag", "DQpos_flag", "EXP_flag", "high_rating"]:
        cs = cs.merge(city_scores(d, c)[["city", f"{c}_raw", f"{c}_se"]],
                      on="city")
    adj, mod = adjusted_city_index(d, "H_any")
    cs["H_adj"] = cs.city.map(adj)
    cs = cs.sort_values("H_any_raw", ascending=False)
    cs.to_csv(f"{OUT}/city_scores.csv", index=False)
    print(cs[["city", "country", "n_reviews", "n_suppliers", "H_any_raw",
              "H_any_lo", "H_any_hi", "H_adj"]].round(3).to_string(index=False))  # noqa: E501

    rho = stats.spearmanr(cs.H_any_raw, cs.H_adj)
    print(f"\nraw vs composition-adjusted city ranking: rho = {rho[0]:.3f} "
          f"(p = {rho[1]:.2g})")

    print("\n" + "=" * 72)
    print("RQ1  VALIDITY")
    print("=" * 72)
    print("\nconvergent -- association with the star rating "
          "(supplier fixed effects, cluster-robust by supplier)")
    dd = d.dropna(subset=["years_ago"])
    Xr = ["H_conduct", "H_ambience", "H_development", "SQpos_flag",
          "SQneg_flag", "DQpos_flag", "DQneg_flag", "EXP_flag", "log_len",
          "translated", "is_creative", "years_ago"]
    tab, info = fe_ols(dd, "rating", Xr, "place_id")
    print(tab.round(4).to_string())
    print(f"  N = {info['N']:,}  suppliers = {info['groups']:,}  "
          f"within R2 = {info['within_r2']:.4f}")

    print("\ndiscriminant -- city-level correlations between constructs")
    cols = ["H_any_raw", "H_conduct_raw", "H_ambience_raw",
            "H_development_raw", "SQpos_flag_raw", "DQpos_flag_raw",
            "EXP_flag_raw", "high_rating_raw"]
    print(cs[cols].corr(method="spearman").round(3).to_string())

    print("\n" + "=" * 72)
    print("RQ2  SERVICE QUALITY AND DESTINATION QUALITY")
    print("=" * 72)
    print("\nreview level -- linear probability model for DQ mention, "
          "supplier fixed effects")
    Xd = ["SQpos_flag", "SQneg_flag", "H_conduct", "H_ambience",
          "H_development", "EXP_flag", "log_len", "translated",
          "is_creative", "years_ago"]
    tab2, info2 = fe_ols(dd, "DQpos_flag", Xd, "place_id")
    print(tab2.round(4).to_string())
    print(f"  N = {info2['N']:,}  suppliers = {info2['groups']:,}  "
          f"within R2 = {info2['within_r2']:.4f}")

    print("\n  same model without supplier fixed effects (city fixed effects)")
    tab2b, info2b = fe_ols(dd, "DQpos_flag", Xd, "city")
    print(tab2b.round(4).to_string())

    print("\nDQ valence given that the destination is mentioned")
    sub = dd[(dd.DQpos_flag + dd.DQneg_flag) > 0].copy()
    sub["dq_pos"] = (sub.DQpos_n >= sub.DQneg_n).astype(int)
    m3 = smf.logit("dq_pos ~ SQpos_flag + SQneg_flag + H_conduct + "
                   "H_ambience + H_development + log_len + translated + "
                   "is_creative", data=sub).fit(disp=0)
    print(pd.DataFrame({"coef": m3.params, "OR": np.exp(m3.params),
                        "p": m3.pvalues}).round(4).to_string())

    print("\ncity level (n = 30)")
    for a, b in [("SQpos_flag_raw", "DQpos_flag_raw"),
                 ("H_any_raw", "DQpos_flag_raw"),
                 ("SQpos_flag_raw", "H_any_raw"),
                 ("SQneg_flag_raw", "DQpos_flag_raw")]:
        r = stats.spearmanr(cs[a], cs[b])
        print(f"  {a:18s} ~ {b:18s} rho = {r[0]:+.3f}  p = {r[1]:.4f}")

    cs.to_csv(f"{OUT}/city_scores.csv", index=False)
    pd.DataFrame(rel, index=["r_sb", "lo", "hi"]
                 ).T.to_csv(f"{OUT}/reliability.csv")
    pd.DataFrame(disc, index=["between_var", "p_perm", "icc"]).T.to_csv(
        f"{OUT}/discrimination.csv")
    print("\nsaved city_scores.csv, reliability.csv, discrimination.csv")


if __name__ == "__main__":
    main()
