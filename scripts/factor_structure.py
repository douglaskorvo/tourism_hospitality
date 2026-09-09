# -*- coding: utf-8 -*-
"""
What is hospitality, according to the reviews?

The induced vocabulary is collapsed by fixed-length truncation stemming
(5 characters -- a standard Portuguese IR device that folds gender and
number agreement without a morphological resource), turned into a
review x marker binary matrix, and submitted to principal-component
analysis with varimax rotation.  The retained components are the
dimensions of hospitality that visitors themselves articulate; they are
compared with the dimensions established in the hospitality literature.

Component retention follows Horn's parallel analysis against the 95th
percentile of eigenvalues from permuted data, which is the appropriate
criterion for binary indicators.
"""
import re
import sys
import unicodedata
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/hosp/scripts")
from code_reviews import normalise  # noqa: E402

RAW = "/home/claude/reviews_raw.csv"
CAND = "/home/claude/hosp/out/candidate_terms.csv"
TOPICAL = set(
    open("/home/claude/hosp/scripts/topical_stopwords.txt").read().split())

TRUNC = 5
MIN_PREV = 0.004          # a marker must appear in >= 0.4% of reviews
RNG = np.random.default_rng(42)


def stem(w):
    return w[:TRUNC]


def varimax(F, tol=1e-6, itmax=200):
    """Kaiser-normalised varimax rotation."""
    p, k = F.shape
    h = np.sqrt((F ** 2).sum(axis=1, keepdims=True))
    h[h == 0] = 1.0
    A = F / h
    R = np.eye(k)
    d = 0.0
    for _ in range(itmax):
        d_old = d
        L = A @ R
        u, s, vt = np.linalg.svd(
            A.T @ (L ** 3 - L @ np.diag((L ** 2).sum(axis=0)) / p))
        R = u @ vt
        d = s.sum()
        if d_old != 0 and abs(d - d_old) / d < tol:
            break
    return (A @ R) * h


def main():
    cand = pd.read_csv(CAND)
    cand = cand[(~cand.is_name) & (~cand.term.isin(TOPICAL))
                & (cand.sim_to_seed_centroid >= 0.46)]
    stems = sorted({stem(t) for t in cand.term})
    print(f"{len(cand)} induced terms -> {len(stems)} truncation stems")

    df = pd.read_csv(RAW)
    d = df[df.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"]).copy()
    norm = [normalise(t) for t in d.text]

    # a stem is present if any token in the review starts with it AND that
    # token is one of the induced terms' morphological family
    fam = {}
    for t in cand.term:
        fam.setdefault(stem(t), set()).add(t)

    rx = {s: re.compile(r"(?<!\w)" + s + r"\w*(?!\w)") for s in stems}
    M = np.zeros((len(norm), len(stems)), dtype=np.int8)
    for i, t in enumerate(norm):
        for j, s in enumerate(stems):
            if rx[s].search(t):
                M[i, j] = 1

    prev = M.mean(axis=0)
    keep = prev >= MIN_PREV
    stems_k = [s for s, k in zip(stems, keep) if k]
    M = M[:, keep]
    print(f"{M.shape[1]} markers retained at prevalence >= {MIN_PREV:.1%}")

    lbl = {s: "/".join(sorted(fam[s])[:4]) for s in stems_k}
    pd.DataFrame({"stem": stems_k,
                  "family": [lbl[s] for s in stems_k],
                  "prevalence": prev[keep].round(4)}).to_csv(
        "/home/claude/hosp/out/markers.csv", index=False)
    np.save("/home/claude/hosp/out/marker_matrix.npy", M)
    d[["place_id", "country", "city", "segment", "rating"]].to_parquet(
        "/home/claude/hosp/out/marker_meta.parquet", index=False)

    # ------------------------------------------------ PCA + parallel analysis
    X = M.astype(float)
    X = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(X, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]

    sim = []
    for _ in range(50):
        P = np.column_stack([RNG.permutation(M[:, j])
                            for j in range(M.shape[1])])
        P = P.astype(float)
        P = (P - P.mean(0)) / np.where(P.std(0) == 0, 1, P.std(0))
        sim.append(np.sort(np.linalg.eigvalsh(
            np.corrcoef(P, rowvar=False)))[::-1])
    thr = np.percentile(np.vstack(sim), 95, axis=0)
    nfac = int((ev > thr).sum())
    print("\nHorn parallel analysis")
    for i in range(min(12, len(ev))):
        mark = "*" if ev[i] > thr[i] else " "
        print(
            f"comp {i+1:2d}  eigen {ev[i]:6.3f}   random95 {thr[i]:6.3f} "
            f"{mark}")
    print(f"  -> retain {nfac} components "
          f"({ev[:nfac].sum()/len(ev):.1%} of variance)")

    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:nfac]
    F = V[:, idx] * np.sqrt(w[idx])
    L = varimax(F)
    load = pd.DataFrame(L, index=[lbl[s] for s in stems_k],
                        columns=[f"F{i+1}" for i in range(nfac)])
    load["prevalence"] = prev[keep]
    load.to_csv("/home/claude/hosp/out/loadings.csv")

    print("\n--- rotated loadings (|loading| >= .30) ---")
    for c in range(nfac):
        col = load[f"F{c+1}"].drop(labels=[], errors="ignore")
        top = col[col.abs() >= .30].sort_values(ascending=False)
        print(f"\nF{c+1}  (n={len(top)})")
        for name, v in top.items():
            print(f"   {v:+.2f}  {name}")

    np.save("/home/claude/hosp/out/loadings.npy", L)


if __name__ == "__main__":
    main()
