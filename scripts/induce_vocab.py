# -*- coding: utf-8 -*-
"""
Bottom-up induction of the hospitality vocabulary from the review corpus.

Rather than imposing an a-priori dictionary, we start from a minimal set of
seed terms whose hospitality meaning is not in dispute ('hospitaleiro',
'acolhedor', 'hospitalidade') and let the corpus tell us which other words
travel with them.  A skip-gram model is trained on the 50,180 Portuguese
reviews; candidate markers are the terms closest to the seed centroid.
The candidate list is then audited by hand and clustered, so the resulting
typology of hospitality is grounded in what visitors actually wrote.
"""
import re
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

sys.path.insert(0, "/home/claude/hosp/scripts")
from code_reviews import normalise  # noqa: E402

RAW = "/home/claude/reviews_raw.csv"
MODEL = "/home/claude/hosp/out/w2v.model"

# minimal, theory-free seeds: the Portuguese words that lexicalise
# 'hospitality' / 'to welcome a guest' and nothing else
SEEDS = ["hospitaleiro", "hospitaleira", "hospitalidade",
         "acolhedor", "acolhedora", "acolhimento", "acolhida"]

STOP = set("""
a o as os um uma uns umas de do da dos das e ou que se com sem por
para pra pelo pela pelos pelas em no na nos nas ao aos eh foi eram era ser sao
muito mais menos ja nao tudo todo toda todos todas eu ele ela eles elas meu
minha nosso nossa nossos nossas seu sua seus suas isso isto esse essa este esta
aquele aquela quando onde como qual quais tem ter tinha havia mas tambem so la
aqui ali entao depois antes durante ate desde nada algum alguma alguns algumas
outro outra outros outras cada mesmo mesma vez vezes ano anos hora horas dia
dias ficou fica ficar foram fomos fui vamos vai ir fazer faz feito fez estava
estavam estao pode podem poder deu dar nos me te lhe seu num numa dele dela
deles delas ns""".split()) | {"|", ","}


def tokenise(t):
    return [w for w in t.split() if w not in STOP and len(w) > 2]


def main():
    df = pd.read_csv(RAW)
    d = df[df.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"])
    print(f"training corpus: {len(d):,} reviews", flush=True)

    sents = [tokenise(normalise(t)) for t in d.text]
    model = Word2Vec(sents, vector_size=200, window=6, min_count=25,
                     sg=1, negative=10, epochs=8, workers=2, seed=42)
    model.save(MODEL)
    print("vocab size:", len(model.wv), flush=True)

    present = [s for s in SEEDS if s in model.wv]
    print("seeds present:", present)
    centroid = np.mean([model.wv[s] for s in present], axis=0)

    sims = model.wv.similar_by_vector(centroid, topn=400)
    rows = [(w, round(float(s), 3), int(model.wv.get_vecattr(w, "count")))
            for w, s in sims]
    out = pd.DataFrame(rows, columns=["term", "sim_to_seed_centroid", "freq"])
    out.to_csv("/home/claude/hosp/out/candidate_terms.csv", index=False)

    print("\n--- top 200 candidate hospitality terms ---")
    for i in range(0, 200, 5):
        print("  " + " | ".join(
            f"{r.term}({r.sim_to_seed_centroid})"
            for r in out.iloc[i:i + 5].itertuples()))


if __name__ == "__main__":
    main()
