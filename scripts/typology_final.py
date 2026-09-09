# -*- coding: utf-8 -*-
"""
An empirical typology of hospitality induced from the reviews.

Marker groups (agreement variants already folded) are represented by the
mean skip-gram vector of their surface forms and clustered with k-means on
the unit sphere.  Two criteria decide k: the silhouette width, and cluster
stability measured as the mean adjusted Rand index across 200 bootstrap
resamples of the marker set.  A typology that does not survive resampling
is not a typology.

The resulting types are then confronted with the dimensions proposed in the
hospitality literature.
"""
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUT = "/home/claude/hosp/out"
RNG = np.random.default_rng(42)
NBOOT = 200


def main():
    model = Word2Vec.load(f"{OUT}/w2v.model")
    mk = pd.read_csv(f"{OUT}/markers.csv")

    vecs, labels = [], []
    for r in mk.itertuples():
        forms = [f for f in r.family.split("/") if f in model.wv]
        if not forms:
            continue
        v = np.mean([model.wv[f] for f in forms], axis=0)
        vecs.append(v / np.linalg.norm(v))
        labels.append(r.family)
    X = np.vstack(vecs)
    print(f"clustering {len(labels)} marker groups")

    stats = {}
    for k in range(3, 10):
        km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        aris = []
        for _ in range(NBOOT):
            idx = RNG.choice(len(X), len(X), replace=True)
            uniq = np.unique(idx)
            kb = KMeans(n_clusters=k, n_init=10,
                        random_state=int(RNG.integers(1e6))).fit(X[idx])
            # relabel the unique members and compare with the full solution
            lab_b = {}
            for pos, i in enumerate(idx):
                lab_b.setdefault(i, kb.labels_[pos])
            aris.append(adjusted_rand_score(km.labels_[uniq],
                                            [lab_b[i] for i in uniq]))
        stats[k] = (sil, float(np.mean(aris)))
        print(
            f"  k={k}  silhouette {sil:.3f}   bootstrap ARI {np.mean(aris):.3f}")  # noqa: E501

    # prefer the most stable solution; break ties by silhouette
    best_k = max(stats, key=lambda k: (round(stats[k][1], 2), stats[k][0]))
    print(f"\nselected k = {best_k}  "
          f"(ARI {stats[best_k][1]:.3f}, silhouette {stats[best_k][0]:.3f})")

    km = KMeans(n_clusters=best_k, n_init=200, random_state=42).fit(X)
    cent = km.cluster_centers_
    centrality = [float(X[i] @ cent[km.labels_[i]] /
                        np.linalg.norm(cent[km.labels_[i]]))
                  for i in range(len(X))]

    out = mk.iloc[:len(labels)].copy()
    out["family"] = labels
    out["type"] = km.labels_
    out["centrality"] = np.round(centrality, 3)
    out = out.sort_values(["type", "centrality"], ascending=[True, False])
    out.to_csv(f"{OUT}/typology.csv", index=False)

    for c in range(best_k):
        sub = out[out.type == c]
        print(f"\n=== TYPE {c}  ({len(sub)} markers, "
              f"total prevalence {sub.prevalence.sum():.1%}) ===")
        for r in sub.itertuples():
            print(f"   {r.prevalence:6.3f}  {r.family[:70]}")


if __name__ == "__main__":
    main()
