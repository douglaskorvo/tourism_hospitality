# -*- coding: utf-8 -*-
"""
Robustness and discriminant validity of the destination hospitality index.

The central threat is that a city-level count of hospitality mentions
measures how EXPRESSIVE a review culture is rather than how hospitable a
destination is: at city level the raw index correlates .92 with generic
experiential affect and .81 with the share of five-star ratings.  Four
tests are run against that threat.

R1  translation      -- Brazilian cities are read in the original language
                        while every other city is machine-translated; the
                        ranking is recomputed on the translated subsample
                        only, so that all cities pass through the same
                        translation channel.
R2  expressiveness   -- the index is re-estimated net of generic affect and
                        of the star rating, and the ranking is compared.
R3  profile          -- the composition of hospitality (share of each type
                        among hospitality-bearing reviews) is by
                        construction free of the overall expressiveness of
                        the corpus.
R4  platform / scope -- China is dropped (Google Maps is not the domestic
                        platform there), and the segment split is tested.

RQ2 is re-examined leaving out one country at a time.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "/home/claude/hosp/scripts")
from analysis import load, city_scores, split_half, between_city_test  # noqa: E402
from fastfe import fe_ols  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "/home/claude/hosp/out"


def city_fe(d, outcome, extra=""):
    """City fixed effects on the logit scale, centred, net of composition."""
    f = (f"{outcome} ~ C(city) + is_creative + log_len + translated + "
         f"years_ago" + extra)
    m = smf.logit(f, data=d.dropna(subset=["years_ago"])).fit(disp=0)
    base = sorted(d.city.unique())[0]
    eff = {base: 0.0}
    for k, v in m.params.items():
        if k.startswith("C(city)[T."):
            eff[k.split("C(city)[T.")[1].rstrip("]")] = v
    s = pd.Series(eff)
    return s - s.mean()


def main():
    d = load()
    cs = pd.read_csv(f"{OUT}/city_scores.csv")
    base_rank = cs.set_index("city").H_any_raw
    res = {}

    print("=" * 74)
    print("R1  TRANSLATION CHANNEL")
    print("=" * 74)
    dt = d[d.translated == 1]
    print(f"translated-only subsample: {len(dt):,} reviews, "
          f"{dt.city.nunique()} cities "
          f"(Brazilian cities keep only their translated minority)")
    tr = dt.groupby("city", observed=True).H_any.mean()
    common = [c for c in base_rank.index if c in tr.index]
    r = stats.spearmanr(base_rank[common], tr[common])
    print(f"rank correlation full vs translated-only: rho = {r[0]:.3f} "
          f"(p = {r[1]:.2g}, n = {len(common)})")
    br = [c for c in common if cs.set_index('city').country[c] == "Brazil"]
    print("\nBrazilian cities, full vs translated-only prevalence:")
    for c in br:
        print(f"   {c:18s} {base_rank[c]:.3f} -> {tr[c]:.3f} "
              f"(n = {(dt.city == c).sum():,})")
    res["R1_rho"] = r[0]

    print("\n" + "=" * 74)
    print("R2  EXPRESSIVENESS")
    print("=" * 74)
    raw = city_fe(d, "H_any")
    net_exp = city_fe(d, "H_any", " + EXP_flag")
    net_both = city_fe(d, "H_any", " + EXP_flag + C(rating)")
    tab = pd.DataFrame({"adj": raw, "net_affect": net_exp,
                        "net_affect_rating": net_both})
    tab["raw_prev"] = base_rank
    tab = tab.sort_values("net_affect_rating", ascending=False)
    print(tab.round(3).to_string())
    print("\nrank correlations")
    for a, b in [("adj", "net_affect"), ("adj", "net_affect_rating"),
                 ("raw_prev", "net_affect_rating")]:
        rr = stats.spearmanr(tab[a], tab[b])
        print(f"   {a:18s} ~ {b:18s} rho = {rr[0]:+.3f}  p = {rr[1]:.2g}")
    res["R2_rho"] = stats.spearmanr(tab["raw_prev"],
                                    tab["net_affect_rating"])[0]
    tab.to_csv(f"{OUT}/city_index_variants.csv")

    print("\n" + "=" * 74)
    print("R3  HOSPITALITY PROFILE (composition, free of expressiveness)")
    print("=" * 74)
    h = d[d.H_any == 1]
    prof = h.groupby("city", observed=True)[
        ["H_conduct", "H_ambience", "H_development"]].mean()
    prof = prof.div(prof.sum(axis=1), axis=0)
    prof["country"] = cs.set_index("city").country
    prof = prof.sort_values("H_development", ascending=False)
    print(prof.round(3).to_string())

    # is the profile itself reliable and discriminating?
    for c in ["H_conduct", "H_ambience", "H_development"]:
        rel = split_half(h, c, reps=150)
        obs, p, icc = between_city_test(h, c)
        print(f"  within hospitality-bearing reviews: {c:15s} "
              f"r_sb = {rel[0]:.3f}  perm p = {p:.4f}  ICC = {icc:.3f}")
    prof.to_csv(f"{OUT}/city_profiles.csv")

    print("\ncorrelation of profile shares with expressiveness and rating")
    aux = prof.join(cs.set_index("city")[["EXP_flag_raw", "high_rating_raw"]])
    for c in ["H_conduct", "H_ambience", "H_development"]:
        r1 = stats.spearmanr(aux[c], aux.EXP_flag_raw)
        r2 = stats.spearmanr(aux[c], aux.high_rating_raw)
        print(f"   {c:15s} ~ affect rho = {r1[0]:+.3f} (p={r1[1]:.3f}) | "
              f"~ 5-star rho = {r2[0]:+.3f} (p={r2[1]:.3f})")

    print("\n" + "=" * 74)
    print("R4  PLATFORM COVERAGE AND SEGMENT")
    print("=" * 74)
    dn = d[d.country != "China"]
    nr = dn.groupby("city", observed=True).H_any.mean()
    cc = [c for c in base_rank.index if c in nr.index]
    print(f"excluding China: {len(dn):,} reviews, {len(cc)} cities, "
          f"rank rho vs full = "
          f"{stats.spearmanr(base_rank[cc], nr[cc])[0]:.3f}")

    for seg in ["mass_market", "creative"]:
        g = d[d.segment == seg].groupby("city", observed=True).H_any.mean()
        cc2 = [c for c in base_rank.index if c in g.index]
        print(f"  {seg:12s} n = {(d.segment == seg).sum():6,}  "
              f"rank rho vs full = "
              f"{stats.spearmanr(base_rank[cc2], g[cc2])[0]:.3f}")
    ms = d[d.segment == "mass_market"].groupby(
        "city", observed=True).H_any.mean()
    cr = d[d.segment == "creative"].groupby("city", observed=True).H_any.mean()
    cc3 = sorted(set(ms.index) & set(cr.index))
    print(f"  mass-market vs creative city rankings: rho = "
          f"{stats.spearmanr(ms[cc3], cr[cc3])[0]:.3f}")

    print("\n" + "=" * 74)
    print("RQ2  LEAVE-ONE-COUNTRY-OUT")
    print("=" * 74)
    dd = d.dropna(subset=["years_ago"])
    Xd = ["SQpos_flag", "SQneg_flag", "H_conduct", "H_ambience",
          "H_development", "EXP_flag", "log_len", "translated",
          "is_creative", "years_ago"]
    full, _ = fe_ols(dd, "DQpos_flag", Xd, "place_id")
    print("full sample:")
    print(full.loc[["SQpos_flag", "SQneg_flag", "H_conduct", "H_ambience",
                    "H_development"]].round(4).to_string())
    print("\nleaving out:")
    rows = []
    for ctry in sorted(d.country.unique()):
        sub = dd[dd.country != ctry]
        t, _ = fe_ols(sub, "DQpos_flag", Xd, "place_id")
        rows.append({"omitted": ctry,
                     "SQpos": t.loc["SQpos_flag", "coef"],
                     "SQpos_p": t.loc["SQpos_flag", "p"],
                     "SQneg": t.loc["SQneg_flag", "coef"],
                     "H_conduct": t.loc["H_conduct", "coef"],
                     "H_ambience": t.loc["H_ambience", "coef"],
                     "H_development": t.loc["H_development", "coef"]})
    loo = pd.DataFrame(rows)
    print(loo.round(4).to_string(index=False))
    loo.to_csv(f"{OUT}/loo_rq2.csv", index=False)

    print("\ncity-level SQ ~ DQ, leaving out one country at a time")
    for ctry in sorted(d.country.unique()):
        sub = cs[cs.country != ctry]
        r = stats.spearmanr(sub.SQpos_flag_raw, sub.DQpos_flag_raw)
        print(f"   omit {ctry:22s} rho = {r[0]:+.3f}  p = {r[1]:.3f}  "
              f"(n = {len(sub)})")


if __name__ == "__main__":
    main()
