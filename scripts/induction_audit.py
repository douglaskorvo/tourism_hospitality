# -*- coding: utf-8 -*-
"""
Document the vocabulary induction as an auditable funnel, and test how much
the two retention thresholds matter.

Every induced term is a single token (the skip-gram vocabulary contains no
multiword units), so occurrence can be evaluated by tokenising each review
once and testing set membership, rather than by running one regular
expression per term. That makes a full threshold sweep affordable.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from gensim.models import Word2Vec

sys.path.insert(0, "/home/claude/hosp/scripts")
from code_reviews import normalise  # noqa: E402
from build_markers import name_gazetteer  # noqa: E402

OUT = "/home/claude/hosp/out"
SUPP = "/home/claude/hosp/out/supp"
RAW = "/home/claude/reviews_raw.csv"
TOPICAL = set(
    open("/home/claude/hosp/scripts/topical_stopwords.txt").read().split())

SEEDS_HOSP = ["hospitaleiro", "hospitaleira", "hospitalidade", "acolhedor",
              "acolhedora", "acolhimento", "acolhida"]
SEEDS_PRAISE = ["incrivel", "otimo", "excelente", "maravilhoso", "perfeito",
                "bom", "otima", "fantastico", "sensacional", "recomendo"]
TRUNC = 5
MAIN = (0.42, 0.02, 0.003)


def centroid(model, words):
    present = [w for w in words if w in model.wv]
    v = np.mean([model.wv[w] for w in present], axis=0)
    return v / np.linalg.norm(v), present


def main():
    model = Word2Vec.load(f"{OUT}/w2v.model")
    ch, seeds_present = centroid(model, SEEDS_HOSP)
    cp, praise_present = centroid(model, SEEDS_PRAISE)
    names = name_gazetteer()

    # ---- score the entire vocabulary once -------------------------------
    rows = []
    for w in model.wv.index_to_key:
        v = model.wv[w]
        v = v / np.linalg.norm(v)
        rows.append((w, float(v @ ch), float(v @ cp), len(w),
                     int(model.wv.get_vecattr(w, "count"))))
    voc = pd.DataFrame(rows, columns=["term", "sim_hosp", "sim_praise",
                                      "length", "freq"])
    voc["margin"] = voc.sim_hosp - voc.sim_praise
    voc["is_name"] = voc.term.isin(names)
    voc["is_topical"] = voc.term.isin(TOPICAL)
    voc["group"] = voc.term.str[:TRUNC]
    voc["group_topical"] = voc.group.isin(TOPICAL)

    # ---- tokenise the corpus once ---------------------------------------
    df = pd.read_csv(RAW)
    d = df[df.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"]).copy()
    print(f"corpus {len(d):,} reviews", flush=True)
    tok = [set(normalise(t).split()) for t in d.text]

    def marker_groups(sim_t, mar_t, prev_t):
        """Apply one threshold triple and return the resulting instrument."""
        c = voc[(voc.sim_hosp >= sim_t) & (voc.margin >= mar_t)
                & (voc.length >= 4) & (~voc.is_name) & (~voc.is_topical)
                & (~voc.group_topical) & (~voc.term.isin(SEEDS_PRAISE))]
        fam = c.groupby("group").term.apply(set).to_dict()
        keep, prevalence = {}, {}
        for g, terms in fam.items():
            hit = np.fromiter((not terms.isdisjoint(s) for s in tok),
                              dtype=bool, count=len(tok))
            p = hit.mean()
            if p >= prev_t:
                keep[g] = hit
                prevalence[g] = p
        return c, keep, prevalence

    # ---- the funnel at the reported thresholds --------------------------
    sim_t, mar_t, prev_t = MAIN
    print("\n=== INDUCTION FUNNEL (reported thresholds) ===")
    steps = []
    v = voc.copy()
    steps.append(("Skip-gram vocabulary (min_count 25)", len(v)))
    v1 = v[v.length >= 4]
    steps.append(("Terms of at least four characters", len(v1)))
    v2 = v1[v1.sim_hosp >= sim_t]
    steps.append(
        (f"Cosine similarity to the hospitality centroid >= {sim_t}", len(v2)))
    v3 = v2[v2.margin >= mar_t]
    steps.append(
        (f"Contrastive margin over generic praise >= {mar_t}", len(v3)))
    v4 = v3[~v3.is_name]
    steps.append(("Personal names removed (gazetteer)", len(v4)))
    v5 = v4[(~v4.is_topical) & (~v4.group_topical)
            & (~v4.term.isin(SEEDS_PRAISE))]
    steps.append(("Topical terms removed (declared list)", len(v5)))
    c, keep, prevalence = marker_groups(*MAIN)
    steps.append(
        (f"Grouped by {TRUNC}-character truncation", c.group.nunique()))
    steps.append((f"Marker groups with prevalence >= {prev_t:.1%}", len(keep)))
    for lab, n in steps:
        print(f"  {n:6,}  {lab}")
    pd.DataFrame(steps, columns=["stage", "n"]).to_csv(
        f"{SUPP}/S16_funnel.csv", index=False)

    print(f"\n  seeds present in the vocabulary: {len(seeds_present)}/"
          f"{len(SEEDS_HOSP)} -> {', '.join(seeds_present)}")
    print(f"  praise terms present: {len(praise_present)}/{len(SEEDS_PRAISE)}")
    print(
        f"gazetteer entries: {len(names):,}; declared topical terms: "
        f"{len(TOPICAL)}")

    # reference index at the main setting
    base_groups = keep
    H_base = np.zeros(len(tok), dtype=bool)
    for h in base_groups.values():
        H_base |= h
    d["_H"] = H_base
    base_city = d.groupby("city", observed=True)._H.mean()

    # ---- threshold sensitivity ------------------------------------------
    print("\n=== THRESHOLD SENSITIVITY ===")
    grid = [(0.40, 0.02, 0.003), (0.42, 0.00, 0.003), (0.42, 0.02, 0.003),
            (0.42, 0.05, 0.003), (0.45, 0.02, 0.003), (0.42, 0.02, 0.001),
            (0.42, 0.02, 0.005)]
    res = []
    for s_t, m_t, p_t in grid:
        c2, k2, _ = marker_groups(s_t, m_t, p_t)
        H = np.zeros(len(tok), dtype=bool)
        for h in k2.values():
            H |= h
        d["_H2"] = H
        cty = d.groupby("city", observed=True)._H2.mean()
        rho = stats.spearmanr(base_city, cty[base_city.index])[0]
        is_main = (s_t, m_t, p_t) == MAIN
        res.append({"sim_threshold": s_t, "margin_threshold": m_t,
                    "prevalence_threshold": p_t, "n_terms": len(c2),
                    "n_marker_groups": len(k2),
                    "corpus_prevalence": float(H.mean()),
                    "rank_rho_vs_reported": rho,
                    "setting": "reported" if is_main else ""})
        print(f"  sim>={s_t}  margin>={m_t}  prev>={p_t:.3f}   "
              f"terms {len(c2):4d}  groups {len(k2):3d}  "
              f"prevalence {H.mean():.3f}  rho {rho:.3f}"
              f"{'   <- reported' if is_main else ''}")
    pd.DataFrame(res).to_csv(f"{SUPP}/S16_sensitivity.csv", index=False)

    # ---- what the thresholds actually exclude ---------------------------
    print("\n=== TERMS AT THE MARGIN (nearest misses and closest calls) ===")
    # any token that appears at all as an author name is withheld from the
    # deposited borderline list, even below the gazetteer threshold, so that
    # no personal name is published in the supplementary material
    import collections as _c
    import re as _re
    import unicodedata as _u
    _raw = pd.read_csv(RAW)
    _any_name = set()
    for _a in _raw.author_name.dropna():
        for _p in str(_a).split():
            _p = _re.sub(r"[^A-Za-zÀ-ÿ]", "", _p)
            if len(_p) > 2:
                _any_name.add("".join(c for c in _u.normalize("NFKD", _p)
                                      if not _u.combining(c)).lower())
    near = voc[(voc.length >= 4) & (~voc.is_name) & (~voc.is_topical)
               & (~voc.term.isin(_any_name))]
    below = near[(near.sim_hosp < sim_t) & (near.sim_hosp >= sim_t - 0.03)
                 & (near.margin >= mar_t)].nlargest(20, "sim_hosp")
    print("  just below the similarity cut:",
          ", ".join(f"{r.term}({r.sim_hosp:.2f})" for r in below.itertuples()))
    lowmar = near[(near.sim_hosp >= sim_t) & (near.margin < mar_t)].nlargest(
        20, "sim_hosp")
    print("  above similarity but failing the contrastive margin:",
          ", ".join(f"{r.term}({r.margin:+.2f})" for r in lowmar.itertuples()))
    pd.concat([below.assign(excluded_by="similarity"),
               lowmar.assign(excluded_by="contrastive margin")])[
        ["term", "sim_hosp", "sim_praise", "margin", "freq",
         "excluded_by"]].to_csv(f"{SUPP}/S16_margin_cases.csv", index=False)

    # ---- typology stability table ---------------------------------------
    print("\n=== TYPOLOGY: k SELECTION ===")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    mk = pd.read_csv(f"{OUT}/markers.csv")
    vecs, labels = [], []
    for r in mk.itertuples():
        forms = [f for f in r.family.split("/") if f in model.wv]
        if forms:
            vv = np.mean([model.wv[f] for f in forms], axis=0)
            vecs.append(vv / np.linalg.norm(vv))
            labels.append(r.family)
    X = np.vstack(vecs)
    rng = np.random.default_rng(42)
    krows = []
    for k in range(3, 10):
        km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        aris = []
        for _ in range(200):
            idx = rng.choice(len(X), len(X), replace=True)
            uniq = np.unique(idx)
            kb = KMeans(n_clusters=k, n_init=10,
                        random_state=int(rng.integers(1e6))).fit(X[idx])
            lab_b = {}
            for pos, i in enumerate(idx):
                lab_b.setdefault(i, kb.labels_[pos])
            aris.append(adjusted_rand_score(km.labels_[uniq],
                                            [lab_b[i] for i in uniq]))
        krows.append({"k": k, "bootstrap_ARI": float(np.mean(aris)),
                      "silhouette": float(sil)})
        print(f"k={k}  bootstrap ARI {np.mean(aris):.3f}  silhouette "
              f"{sil:.3f}")
    pd.DataFrame(krows).to_csv(f"{SUPP}/S16_k_selection.csv", index=False)
    print("\nwritten to", SUPP)


if __name__ == "__main__":
    main()
