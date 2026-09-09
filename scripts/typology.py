# -*- coding: utf-8 -*-
"""
Induce an empirical typology of hospitality from the reviews.

The candidate vocabulary produced by induce_vocab.py is clustered in the
skip-gram embedding space.  Clusters are the *types* of hospitality that
visitors actually articulate; they are then confronted with the constructs
established in the hospitality literature.
"""
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL = "/home/claude/hosp/out/w2v.model"
CAND = "/home/claude/hosp/out/candidate_terms.csv"

# terms that are topical (what the business sells, where it is) rather than
# relational (how the guest was treated); removed before clustering
TOPICAL = set("""
playlist ceramista estudio hostel danco planet interiores oficio
prof novato bhaiya carioca rapazes universo auto habitual agitacao daqueles
juntas curtindo aprenderem treinos alunas instrutores mentores retiro terapia
homestay soul epitome dom lar pedacinho cantinho refugio danca aulas escola
equipe proprietarios administradores administrado administradora recepcionistas
anfitrioes anfitria hospede professoras ceramica danco treinado treinada
capacitados qualificados habilidosa talentosa talentosos didatica sabedoria
ensinamentos criatividade estetica decorado arejado chique elegante sofisticado
sofisticada requintada refinado romantico saudavel interativa dinamica
dinamicas
focada focado prepara presta prestando trazem cria criam promove contribui
contribuiu contribuindo demonstram demonstrado transmite transparece distingue
diferencia diferenciam melhorando melhoraram transformaram destacaram sentiu
sentira sentisse sentissem sentissemos sintam sentem sinta vindos bracos
instantaneamente extremo extrema sensacao achado vivi vivencia rir tocou
lembraremos parabenizar satisfazer despertar individuo personalidade atitude
postura sentimento espirito almas verdadeiras tratados tratou trataram
envolvido envolvimento interacoes apreciado amando encantado inspirado
apoiado apoiou ouve novato""".split())


def main():
    model = Word2Vec.load(MODEL)
    cand = pd.read_csv(CAND)
    cand = cand[(~cand.is_name) & (~cand.term.isin(TOPICAL))]
    cand = cand[cand.sim_to_seed_centroid >= 0.46].reset_index(drop=True)
    terms = [t for t in cand.term if t in model.wv]
    X = np.vstack([model.wv[t] for t in terms])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    print(f"clustering {len(terms)} induced hospitality terms")

    scores = {}
    for k in range(4, 12):
        km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    print("silhouette by k:", {k: round(v, 4) for k, v in scores.items()})
    best_k = max(scores, key=scores.get)
    print("selected k =", best_k)

    km = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(X)
    cand = cand[cand.term.isin(terms)].copy()
    cand["cluster"] = km.labels_
    # order terms within a cluster by closeness to its centroid
    cent = km.cluster_centers_
    cand["centrality"] = [float(X[i] @ cent[km.labels_[i]])
                          for i in range(len(terms))]
    cand = cand.sort_values(["cluster", "centrality"], ascending=[True, False])
    cand.to_csv("/home/claude/hosp/out/typology_clusters.csv", index=False)

    for c in range(best_k):
        sub = cand[cand.cluster == c]
        print(f"\n=== cluster {c}  (n={len(sub)}) ===")
        print("  " + ", ".join(sub.term.head(28)))


if __name__ == "__main__":
    main()
