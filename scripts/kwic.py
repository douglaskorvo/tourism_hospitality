# -*- coding: utf-8 -*-
"""Keyword-in-context sampler for lexicon auditing."""
import sys
import re
import random
import pandas as pd
sys.path.insert(0, "/home/claude/hosp/scripts")
from code_reviews import COMPILED, normalise, RE_DHOSP  # noqa: E402

df = pd.read_csv("/home/claude/reviews_raw.csv")
d = df[df.lang.isin(["pt", "pt-BR"])].dropna(subset=["text"])
random.seed(7)
texts = [normalise(t) for t in d.text.sample(6000, random_state=7)]

cons, sub, k = sys.argv[1], sys.argv[2], int(
    sys.argv[3]) if len(sys.argv) > 3 else 25
rx = RE_DHOSP if cons == "DEST" else COMPILED[cons][sub]
hits = []
for t in texts:
    for m in rx.finditer(t):
        a, b = max(0, m.start() - 70), min(len(t), m.end() + 70)
        hits.append(f"...{t[a:m.start()]}[[{m.group(0)}]]{t[m.end():b]}...")
random.shuffle(hits)
print(f"### {cons}/{sub}  total hits in 6k sample: {len(hits)}")
for h in hits[:k]:
    print(" -", h)
