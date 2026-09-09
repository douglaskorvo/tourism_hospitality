# -*- coding: utf-8 -*-
"""
Export every quantity that appears in the manuscript into tidy CSVs, so the
supplementary workbook is assembled from computed values rather than from
numbers retyped out of a log.

Where the workbook will carry a live formula, this script exports the RAW
INPUTS of that formula (counts, not percentages; the half-split correlation,
not the Spearman-Brown value) so the spreadsheet can do the arithmetic and a
reader can audit it.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/home/claude/hosp/scripts")
from analysis import load, between_city_test, rng_for  # noqa: E402
from fastfe import fe_ols  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "/home/claude/hosp/out"
SUPP = "/home/claude/hosp/out/supp"
RNG = np.random.default_rng(2026)
NBOOT = 500

import os  # noqa: E402
os.makedirs(SUPP, exist_ok=True)

CONSTRUCTS = {
    "H_any": "Hospitality, any type",
    "H_conduct": "Hospitality type 1: interpersonal conduct",
    "H_ambience": "Hospitality type 2: created ambience",
    "H_development": "Hospitality type 3: development",
    "DESTHOSP_flag": "Hospitality predicated of the destination itself",
    "SQpos_flag": "Service quality, positive",
    "SQneg_flag": "Service quality, negative",
    "DQpos_flag": "Destination quality, positive",
    "DQneg_flag": "Destination quality, negative",
    "EXP_flag": "Generic experiential affect",
    "INTpos_flag": "Behavioural intention, positive",
    "INTneg_flag": "Behavioural intention, negative",
}


def main():
    d = load()
    d["names_host"] = pd.read_csv(f"{OUT}/named_host.csv").set_index(
        "review_id").reindex(d.review_id).names_host.values

    # ------------------------------------------------ S2 corpus by destination
    g = d.groupby(["country", "city"], observed=True)
    corpus = g.agg(
        suppliers=("place_id", "nunique"),
        reviews=("rating", "size"),
        rating_sum=("rating", "sum"),
        translated_n=("translated", "sum"),
        creative_n=("is_creative", "sum"),
        tokens_sum=("n_tokens", "sum"),
        H_n=("H_any", "sum"),
        H1_n=("H_conduct", "sum"),
        H2_n=("H_ambience", "sum"),
        H3_n=("H_development", "sum"),
        SQpos_n=("SQpos_flag", "sum"),
        SQneg_n=("SQneg_flag", "sum"),
        DQpos_n=("DQpos_flag", "sum"),
        DQneg_n=("DQneg_flag", "sum"),
        EXP_n=("EXP_flag", "sum"),
        named_n=("names_host", "sum"),
    ).reset_index()
    corpus.to_csv(f"{SUPP}/S2_corpus.csv", index=False)

    # ------------------------------------------- S3 review-level prevalence
    rows = []
    for k, lab in CONSTRUCTS.items():
        rows.append({"indicator": k, "label": lab,
                     "n_reviews_flagged": int(d[k].sum()),
                     "n_reviews_total": len(d)})
    pd.DataFrame(rows).to_csv(f"{SUPP}/S3_prevalence.csv", index=False)

    # ------------------------------------------------------- S4 marker set
    mk = pd.read_csv(f"{OUT}/markers.csv")
    typ = pd.read_csv(f"{OUT}/typology.csv")[["family", "type", "centrality"]]
    names = {0: "3 Development",
             1: "1 Interpersonal conduct", 2: "2 Created ambience"}
    mk = mk.merge(typ, on="family", how="left")
    mk["type_label"] = mk.type.map(names)
    mk["n_reviews"] = (mk.prevalence * len(d)).round().astype(int)
    mk["n_total"] = len(d)
    mk = mk.sort_values(["type_label", "prevalence"], ascending=[True, False])
    mk[["family", "group", "type_label", "n_reviews", "n_total",
        "sim_hosp", "margin", "centrality"]].to_csv(
        f"{SUPP}/S4_markers.csv", index=False)

    # ------------------------------------------- S5 city index + components
    rows = []
    for city, gg in d.groupby("city", observed=True):
        agg = gg.groupby("place_id", observed=True)[
            "H_any"].agg(["sum", "size"])
        sv = agg["sum"].values.astype(float)
        nv = agg["size"].values.astype(float)
        idx = rng_for("H_any", "bootstrap" + city).integers(
            0, len(sv), size=(NBOOT, len(sv)))
        boots = sv[idx].sum(axis=1) / nv[idx].sum(axis=1)
        rows.append({
            "city": city, "country": gg.country.iloc[0],
            "n_suppliers": gg.place_id.nunique(),
            "n_reviews": len(gg),
            "n_hosp": int(gg.H_any.sum()),
            "boot_lo": np.percentile(boots, 2.5),
            "boot_hi": np.percentile(boots, 97.5),
            "boot_se": boots.std(ddof=1),
        })
    city = pd.DataFrame(rows)
    var = pd.read_csv(f"{OUT}/city_index_variants.csv", index_col=0)
    city["index_adjusted"] = city.city.map(var["adj"])
    city["index_net_affect"] = city.city.map(var["net_affect"])
    city["index_net_affect_rating"] = city.city.map(var["net_affect_rating"])
    city.to_csv(f"{SUPP}/S5_city_index.csv", index=False)

    # ------------------------------------------------------- S6 city profile
    h = d[d.H_any == 1]
    prof = h.groupby("city", observed=True).agg(
        n_hosp_reviews=("H_any", "size"),
        n_conduct=("H_conduct", "sum"),
        n_ambience=("H_ambience", "sum"),
        n_development=("H_development", "sum")).reset_index()
    prof["country"] = prof.city.map(city.set_index("city").country)
    prof.to_csv(f"{SUPP}/S6_city_profile.csv", index=False)

    # ------------------------------ S7 reliability (raw half correlations)
    def split_half_raw(data, col, reps=200):
        """Same splits as analysis.split_half, reported before the
        Spearman-Brown correction so the workbook can apply it as a formula."""
        rng = rng_for(col, "split_half")
        agg = {}
        for c, gg in data.groupby("city", observed=True):
            a = gg.groupby("place_id", observed=True)[col].agg(["sum", "size"])
            agg[c] = (a["sum"].values.astype(float),
                      a["size"].values.astype(float))
        rs = []
        for _ in range(reps):
            h1, h2 = [], []
            for c in sorted(agg):
                sv, nv = agg[c]
                perm = rng.permutation(len(sv))
                i1, i2 = perm[:len(sv) // 2], perm[len(sv) // 2:]
                if nv[i1].sum() > 20 and nv[i2].sum() > 20:
                    h1.append(sv[i1].sum() / nv[i1].sum())
                    h2.append(sv[i2].sum() / nv[i2].sum())
            if len(h1) > 5:
                rs.append(stats.pearsonr(h1, h2)[0])
        return float(np.mean(rs)), float(np.percentile(rs, 2.5)), \
            float(np.percentile(rs, 97.5))

    rel_rows = []
    for col in ["H_any", "H_conduct", "H_ambience", "H_development",
                "SQpos_flag", "DQpos_flag"]:
        r, lo, hi = split_half_raw(d, col)
        obs, p, icc = between_city_test(d, col)
        rel_rows.append({"indicator": col, "label": CONSTRUCTS[col],
                         "r_half": r, "r_half_lo": lo, "r_half_hi": hi,
                         "between_city_var": obs, "perm_p": p, "icc1": icc})
    pd.DataFrame(rel_rows).to_csv(f"{SUPP}/S7_reliability.csv", index=False)

    # profile reliability, within hospitality-bearing reviews
    prof_rows = []
    for col in ["H_conduct", "H_ambience", "H_development"]:
        r, lo, hi = split_half_raw(h, col, reps=150)
        obs, p, icc = between_city_test(h, col)
        prof_rows.append({"indicator": col + " (share within hospitality reviews)",  # noqa: E501
                          "label": CONSTRUCTS[col], "r_half": r,
                          "r_half_lo": lo, "r_half_hi": hi,
                          "between_city_var": obs, "perm_p": p, "icc1": icc})
    pd.DataFrame(prof_rows).to_csv(f"{SUPP}/S7b_profile_reliability.csv",
                                   index=False)

    # ---------------------------------------------------- S8 rating model
    dd = d.dropna(subset=["years_ago"])
    Xr = ["H_conduct", "H_ambience", "H_development", "SQpos_flag",
          "SQneg_flag", "DQpos_flag", "DQneg_flag", "EXP_flag", "log_len",
          "translated", "is_creative", "years_ago"]
    tab, info = fe_ols(dd, "rating", Xr, "place_id")
    tab = tab.reset_index().rename(columns={"index": "predictor"})
    tab["n"] = info["N"]
    tab["suppliers"] = info["groups"]
    tab["within_r2"] = info["within_r2"]
    tab.to_csv(f"{SUPP}/S8_rating_model.csv", index=False)

    # ------------------------------------------------------- S9 RQ2 models
    Xd = ["SQpos_flag", "SQneg_flag", "H_conduct", "H_ambience",
          "H_development", "EXP_flag", "log_len", "translated",
          "is_creative", "years_ago"]
    a, ia = fe_ols(dd, "DQpos_flag", Xd, "place_id")
    a = a.reset_index().rename(columns={"index": "predictor"})
    a["model"] = "A: supplier fixed effects"
    b, ib = fe_ols(dd, "DQpos_flag", Xd, "city")
    b = b.reset_index().rename(columns={"index": "predictor"})
    b["model"] = "A2: city fixed effects"
    pd.concat([a, b]).to_csv(f"{SUPP}/S9_rq2_modelA.csv", index=False)

    sub = dd[(dd.DQpos_flag + dd.DQneg_flag) > 0].copy()
    sub["dq_pos"] = (sub.DQpos_n >= sub.DQneg_n).astype(int)
    m3 = smf.logit("dq_pos ~ SQpos_flag + SQneg_flag + H_conduct + "
                   "H_ambience + H_development + log_len + translated + "
                   "is_creative", data=sub).fit(disp=0)
    pd.DataFrame({"predictor": m3.params.index, "coef": m3.params.values,
                  "se": m3.bse.values, "odds_ratio": np.exp(m3.params.values),
                  "p": m3.pvalues.values,
                  "n": len(sub)}).to_csv(f"{SUPP}/S9b_rq2_modelB.csv",
                                         index=False)

    # ------------------------------------------ S10 leave-one-country-out
    rows = []
    for ctry in sorted(d.country.unique()):
        s = dd[dd.country != ctry]
        t, i2 = fe_ols(s, "DQpos_flag", Xd, "place_id")
        cs = pd.read_csv(f"{OUT}/city_scores.csv")
        cc = cs[cs.country != ctry]
        rho, pv = stats.spearmanr(cc.SQpos_flag_raw, cc.DQpos_flag_raw)
        rows.append({
            "omitted_country": ctry, "n_reviews": i2["N"],
            "SQpos_coef": t.loc["SQpos_flag", "coef"],
            "SQpos_se": t.loc["SQpos_flag", "se"],
            "SQpos_p": t.loc["SQpos_flag", "p"],
            "SQneg_coef": t.loc["SQneg_flag", "coef"],
            "H_conduct_coef": t.loc["H_conduct", "coef"],
            "H_ambience_coef": t.loc["H_ambience", "coef"],
            "H_development_coef": t.loc["H_development", "coef"],
            "city_rho_SQ_DQ": rho, "city_rho_p": pv, "n_cities": len(cc)})
    pd.DataFrame(rows).to_csv(f"{SUPP}/S10_loo.csv", index=False)

    # -------------------------------------------- S11 city correlations
    cs = pd.read_csv(f"{OUT}/city_scores.csv").set_index("city")
    cols = {"H_any_raw": "Hospitality (any)",
            "H_conduct_raw": "Type 1 conduct",
            "H_ambience_raw": "Type 2 ambience",
            "H_development_raw": "Type 3 development",
            "SQpos_flag_raw": "Service quality +",
            "SQneg_flag_raw": "Service quality -",
            "DQpos_flag_raw": "Destination quality +",
            "EXP_flag_raw": "Generic affect",
            "high_rating_raw": "Share of 5-star"}
    cm = cs[list(cols)].rename(columns=cols).corr(method="spearman")
    cm.to_csv(f"{SUPP}/S11_city_correlations.csv")

    # -------------------------------------------------- S12 robustness
    base = cs.H_any_raw
    rob = []
    dt = d[d.translated == 1]
    tr = dt.groupby("city", observed=True).H_any.mean()
    common = [c for c in base.index if c in tr.index]
    rob.append({"test": "Translated reviews only",
                "addresses": "Brazilian cities are served in the original language",  # noqa: E501
                "n_reviews": len(dt), "n_cities": len(common),
                "rank_rho": stats.spearmanr(base[common], tr[common])[0]})
    rob.append({"test": "Net of generic affect and star rating",
                "addresses": "Index may measure expressiveness of the review culture",  # noqa: E501
                "n_reviews": len(d), "n_cities": len(var),
                "rank_rho": stats.spearmanr(var["adj"],
                                            var["net_affect_rating"])[0]})
    dn = d[d.country != "China"]
    nr = dn.groupby("city", observed=True).H_any.mean()
    cc = [c for c in base.index if c in nr.index]
    rob.append({"test": "Excluding China",
                "addresses": "Google Maps is not the domestic platform in mainland China",  # noqa: E501
                "n_reviews": len(dn), "n_cities": len(cc),
                "rank_rho": stats.spearmanr(base[cc], nr[cc])[0]})
    for seg, lab in [("mass_market", "Mass-market suppliers only"),
                     ("creative", "Independent and creative suppliers only")]:
        gsub = d[d.segment == seg].groupby("city", observed=True).H_any.mean()
        c2 = [c for c in base.index if c in gsub.index]
        rob.append({"test": lab, "addresses": "Supplier-mix composition",
                    "n_reviews": int((d.segment == seg).sum()),
                    "n_cities": len(c2),
                    "rank_rho": stats.spearmanr(base[c2], gsub[c2])[0]})
    pd.DataFrame(rob).to_csv(f"{SUPP}/S12_robustness.csv", index=False)

    # ------------------------------------------- S13 personal attribution
    att = []
    for lab, sub2 in [("All reviews", d),
                      ("Hospitality-bearing", d[d.H_any == 1]),
                      ("No hospitality marker", d[d.H_any == 0]),
                      ("Type 1 conduct", d[d.H_conduct == 1]),
                      ("Type 2 ambience", d[d.H_ambience == 1]),
                      ("Type 3 development", d[d.H_development == 1]),
                      ("Service quality mention", d[d.SQpos_flag == 1]),
                      ("Destination quality mention", d[d.DQpos_flag == 1])]:
        att.append({"subset": lab, "n_reviews": len(sub2),
                    "n_naming_a_person": int(sub2.names_host.sum())})
    pd.DataFrame(att).to_csv(f"{SUPP}/S13_attribution.csv", index=False)

    # ----------------------------------------- S14 discriminant term list
    dt2 = pd.read_csv(f"{OUT}/discriminant_terms.csv")
    dt2 = dt2.sort_values("sim_hosp", ascending=False)
    dt2.to_csv(f"{SUPP}/S14_terms.csv", index=False)

    # ------------------------------------- S15 type co-occurrence
    co = []
    for a1 in ["H_conduct", "H_ambience", "H_development"]:
        for b1 in ["H_conduct", "H_ambience", "H_development"]:
            if a1 < b1:
                co.append({"type_a": CONSTRUCTS[a1], "type_b": CONSTRUCTS[b1],
                           "n_a": int(d[a1].sum()), "n_b": int(d[b1].sum()),
                           "n_both": int((d[a1] & d[b1]).sum()),
                           "n_total": len(d),
                           "phi": float(np.corrcoef(d[a1], d[b1])[0, 1])})
    pd.DataFrame(co).to_csv(f"{SUPP}/S15_cooccurrence.csv", index=False)

    print("supplementary CSVs written to", SUPP)
    import os as _os
    for f in sorted(_os.listdir(SUPP)):
        print("  ", f)


if __name__ == "__main__":
    main()
