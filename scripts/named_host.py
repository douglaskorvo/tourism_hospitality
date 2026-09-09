# -*- coding: utf-8 -*-
"""
Does a review name the person who hosted it?

Reviews are flagged when they mention a first name drawn from the platform's
own author-name field, excluding the reviewer's own name. The gazetteer is
the one built in build_markers.py, so the exclusion list used during
vocabulary induction and the detection list used here cannot diverge.

Feeds Section 4.2.3 of the manuscript and sheet S13 of the supplementary
workbook.
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/home/claude/hosp/scripts")
from code_reviews import normalise  # noqa: E402
from build_markers import name_gazetteer  # noqa: E402

RAW = "/home/claude/reviews_raw.csv"
OUT = "/home/claude/hosp/out"


def strip(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower()


def main():
    gaz = name_gazetteer()
    print(f"gazetteer: {len(gaz):,} unambiguous first-name tokens")

    raw = pd.read_csv(RAW)
    raw["review_id"] = np.arange(len(raw))
    d = raw[raw.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"]).copy()

    rx = re.compile(r"(?<!\w)(" + "|".join(
        sorted((re.escape(g) for g in gaz), key=len, reverse=True)) + r")(?!\w)")  # noqa: E501

    flags = []
    for author, text in zip(d.author_name, d.text):
        own = {strip(p) for p in str(author).split()}
        found = set(rx.findall(normalise(text))) - own
        flags.append(1 if found else 0)
    d["names_host"] = flags

    d[["review_id", "names_host"]].to_csv(f"{OUT}/named_host.csv", index=False)
    print(
        f"reviews naming a person other than the reviewer: "
        f"{np.mean(flags):.1%}")

    # the contrast reported in Section 4.2.3 needs the coded table
    coded = pd.read_parquet(f"{OUT}/reviews_coded.parquet")
    M = np.load(f"{OUT}/marker_matrix.npy")
    typ = pd.read_csv(f"{OUT}/typology.csv")
    mk = pd.read_csv(f"{OUT}/markers.csv")
    order = {f: i for i, f in enumerate(mk.family)}
    cols = [order[f] for f in typ.family if f in order]
    coded["H_any"] = (M[:, cols].sum(axis=1) > 0).astype(int)
    coded["names_host"] = d.names_host.values

    ct = pd.crosstab(coded.H_any, coded.names_host)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / len(coded))
    print(
        f"  hospitality-bearing : {coded[coded.H_any == 1].names_host.mean():.1%}")  # noqa: E501
    print(
        f"  no hospitality      : {coded[coded.H_any == 0].names_host.mean():.1%}")  # noqa: E501
    print(f"  chi2 = {chi2:.1f}, p = {p:.3g}, Cramer's V = {v:.3f}")


if __name__ == "__main__":
    main()
