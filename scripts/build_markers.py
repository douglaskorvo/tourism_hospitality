# -*- coding: utf-8 -*-
"""
Discriminant induction of the hospitality marker set.

Two problems have to be solved before the vocabulary can be used.

(1) Seed expansion in embedding space also retrieves generic praise
    ('incrivel', 'maravilhoso', 'excepcional'), because enthusiastic reviews
    contain both.  We therefore score every candidate CONTRASTIVELY: its
    cosine similarity to the hospitality seed centroid minus its similarity
    to a generic-praise centroid.  Only terms that are closer to
    hospitality than to praise are retained.  This is what separates the
    construct from the halo that surrounds it.

(2) Truncation stemming folds agreement but also collides unrelated words
    ('trans' -> transformadora / transporte / transparente).  Stems are
    therefore used only as GROUP LABELS; occurrence is matched against the
    explicit list of induced surface forms, never against a prefix.
"""
import collections
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

sys.path.insert(0, "/home/claude/hosp/scripts")
from code_reviews import normalise  # noqa: E402

OUT = "/home/claude/hosp/out"
MODEL = f"{OUT}/w2v.model"
RAW = "/home/claude/reviews_raw.csv"
TOPICAL = set(
    open("/home/claude/hosp/scripts/topical_stopwords.txt").read().split())

SEEDS_HOSP = ["hospitaleiro", "hospitaleira", "hospitalidade", "acolhedor",
              "acolhedora", "acolhimento", "acolhida"]
SEEDS_PRAISE = ["incrivel", "otimo", "excelente", "maravilhoso", "perfeito",
                "bom", "otima", "fantastico", "sensacional", "recomendo"]

MIN_SIM = 0.42          # closeness to the hospitality centroid
MIN_MARGIN = 0.02       # must be closer to hospitality than to generic praise
MIN_PREV = 0.003        # marker must occur in >= 0.3% of reviews
TRUNC = 5               # characters used to group agreement variants


def name_gazetteer(min_author_freq=15, word_ratio=8):
    """Personal names, built from the platform's own author-name field.

    A token qualifies only if it is common as an author name AND is not
    markedly more frequent as an ordinary word, which keeps Portuguese nouns
    that happen to be spelled like names ('rosa', 'gloria', 'linda') out of
    the exclusion list. The same gazetteer is used for the personal
    attribution analysis, so the two are guaranteed to agree.
    """
    def strip(s):
        return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                       if not unicodedata.combining(c)).lower()

    raw = pd.read_csv(RAW)
    namef = collections.Counter()
    for a in raw.author_name.dropna():
        for part in str(a).split():
            part = re.sub(r"[^A-Za-zÀ-ÿ]", "", part)
            if len(part) > 2:
                namef[strip(part)] += 1

    d = raw[raw.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"])
    textf = collections.Counter()
    for t in d.text:
        textf.update(w for w in normalise(t).split() if len(w) > 2)

    return {n for n, k in namef.items()
            if k >= min_author_freq and textf.get(n, 0) <= word_ratio * k}


def centroid(model, words):
    v = np.mean([model.wv[w] for w in words if w in model.wv], axis=0)
    return v / np.linalg.norm(v)


def main():
    model = Word2Vec.load(MODEL)
    ch = centroid(model, SEEDS_HOSP)
    cp = centroid(model, SEEDS_PRAISE)

    names = name_gazetteer()
    print(f"personal-name gazetteer: {len(names):,} tokens")

    rows = []
    for w in model.wv.index_to_key:
        if len(w) < 4 or w in names or w in TOPICAL or w in SEEDS_PRAISE:
            continue
        v = model.wv[w]
        v = v / np.linalg.norm(v)
        sh, sp = float(v @ ch), float(v @ cp)
        if sh >= MIN_SIM and (sh - sp) >= MIN_MARGIN:
            rows.append((w, sh, sp, sh - sp,
                         int(model.wv.get_vecattr(w, "count"))))
    cand = pd.DataFrame(rows, columns=["term", "sim_hosp", "sim_praise",
                                       "margin", "freq"])
    cand = cand.sort_values("sim_hosp", ascending=False).reset_index(drop=True)
    print(f"{len(cand)} discriminant hospitality terms "
          f"(sim>={MIN_SIM}, margin>={MIN_MARGIN})")

    cand["group"] = cand.term.str[:TRUNC]
    # a group whose key is itself a topical word is dropped as a whole, so
    # that inflected forms cannot re-enter through the back door
    cand = cand[~cand.group.isin(TOPICAL)]
    fam = cand.groupby("group").term.apply(list).to_dict()

    # ---- occurrence matrix, matched on explicit surface forms -------------
    df = pd.read_csv(RAW)
    d = df[df.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"]).copy()
    norm = [normalise(t) for t in d.text]

    groups = sorted(fam)
    rx = {g: re.compile("|".join(r"(?<!\w)%s(?!\w)" % re.escape(t)
                                 for t in fam[g])) for g in groups}
    M = np.zeros((len(norm), len(groups)), dtype=np.int8)
    for i, t in enumerate(norm):
        for j, g in enumerate(groups):
            if rx[g].search(t):
                M[i, j] = 1

    prev = M.mean(axis=0)
    keep = prev >= MIN_PREV
    groups_k = [g for g, k in zip(groups, keep) if k]
    M = M[:, keep]
    print(
        f"{M.shape[1]} marker groups retained at prevalence >= {MIN_PREV:.1%}")

    mk = pd.DataFrame({
        "group": groups_k,
        "family": ["/".join(sorted(fam[g])) for g in groups_k],
        "prevalence": prev[keep].round(4),
        "sim_hosp": [cand[cand.group == g].sim_hosp.max() for g in groups_k],
        "margin": [cand[cand.group == g].margin.max() for g in groups_k],
    }).sort_values("prevalence", ascending=False)
    mk.to_csv(f"{OUT}/markers.csv", index=False)
    np.save(f"{OUT}/marker_matrix.npy", M)
    d[["place_id", "country", "city", "segment", "rating", "lang"]].to_parquet(
        f"{OUT}/marker_meta.parquet", index=False)
    cand.to_csv(f"{OUT}/discriminant_terms.csv", index=False)

    print("\n--- marker groups by prevalence ---")
    for r in mk.itertuples():
        print(f"  {r.prevalence:6.3f}  {r.family[:78]}")


if __name__ == "__main__":
    main()
