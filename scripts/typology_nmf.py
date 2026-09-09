# -*- coding: utf-8 -*-
"""
An empirical typology of hospitality, induced from the reviews.

The review x marker matrix (markers = the induced hospitality vocabulary,
folded by 5-character truncation stemming) is factorised by non-negative
matrix factorisation on TF-IDF weights.  NMF is preferred to PCA here
because the data are sparse, non-negative and binary, and because its parts
based decomposition yields additive, directly interpretable types rather
than bipolar contrasts.

The number of types is chosen by NPMI topic coherence, computed on the same
corpus, averaged over the top markers of each type; ties are broken towards
the more parsimonious solution.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

OUT = "/home/claude/hosp/out"
TOPN = 10


def npmi_coherence(M, top_idx):
    """Average NPMI over all marker pairs in a type."""
    n = M.shape[0]
    sc = []
    for a in range(len(top_idx)):
        for b in range(a + 1, len(top_idx)):
            i, j = top_idx[a], top_idx[b]
            pi = M[:, i].mean()
            pj = M[:, j].mean()
            pij = (M[:, i] & M[:, j]).mean()
            if pij == 0 or pi == 0 or pj == 0:
                sc.append(-1.0)
                continue
            pmi = np.log(pij / (pi * pj))
            sc.append(pmi / -np.log(pij))
    return float(np.mean(sc))


def main():
    M = np.load(f"{OUT}/marker_matrix.npy")
    mk = pd.read_csv(f"{OUT}/markers.csv")
    labels = mk.family.tolist()

    # restrict to reviews that actually carry hospitality content
    has = M.sum(axis=1) > 0
    print(f"{has.sum():,} of {len(M):,} reviews carry >=1 hospitality marker "
          f"({has.mean():.1%})")
    Mh = M[has]

    tf = TfidfTransformer(sublinear_tf=False).fit_transform(Mh)

    res = {}
    for k in range(3, 11):
        nmf = NMF(n_components=k, init="nndsvda", random_state=42,
                  max_iter=600, l1_ratio=0.3, alpha_W=0.0)
        W = nmf.fit_transform(tf)
        H = nmf.components_
        cohs = [npmi_coherence(Mh.astype(bool),
                               list(np.argsort(H[c])[::-1][:TOPN]))
                for c in range(k)]
        res[k] = (float(np.mean(cohs)), nmf.reconstruction_err_, W, H)
        print(f"  k={k:2d}  NPMI coherence {np.mean(cohs):+.4f}   "
              f"recon.err {nmf.reconstruction_err_:.2f}")

    best_k = max(res, key=lambda k: res[k][0])
    print(f"\nselected k = {best_k} (highest NPMI coherence)")
    coh, err, W, H = res[best_k]

    rows = []
    for c in range(best_k):
        order = np.argsort(H[c])[::-1][:14]
        rows.append({"type": c + 1,
                     "share_of_reviews": float((W.argmax(axis=1) == c).mean()),
                     "markers": ", ".join(labels[i] for i in order),
                     "weights": ", ".join(f"{H[c][i]:.2f}" for i in order)})
        print(f"\n=== TYPE {c+1}  ({(W.argmax(axis=1)==c).mean():.1%} of "
              f"hospitality-bearing reviews) ===")
        print("   " + ", ".join(labels[i] for i in order))
    pd.DataFrame(rows).to_csv(f"{OUT}/hospitality_types.csv", index=False)

    # review-level type scores, projected back to the full corpus
    Wfull = np.zeros((len(M), best_k))
    Wfull[has] = W
    np.save(f"{OUT}/type_scores.npy", Wfull)
    np.save(f"{OUT}/type_components.npy", H)
    print("\nsaved type scores", Wfull.shape)


if __name__ == "__main__":
    main()
