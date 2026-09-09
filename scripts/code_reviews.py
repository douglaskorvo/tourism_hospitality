# -*- coding: utf-8 -*-
"""
Dictionary-based prevalence coding of Google Maps reviews for
destination-level hospitality analysis (attribution-first coding).

Pipeline
--------
1. restrict the corpus to text served in Portuguese ('pt' = original,
   'pt-BR' = machine-translated by the platform), the language in which the
   Places API returned reviews for all 30 destinations -- this is what makes
   a single lexicon comparable across countries;
2. normalise (casefold, strip diacritics, drop punctuation);
3. match construct stems with word boundaries;
4. apply a 3-token left negation window (polarity flip);
5. resolve the referent of every hit as provider / destination / none using
   the nearest referent anchor inside a +/-12-token window;
6. credit AMBIGUOUS markers to a construct only when the resolved referent
   matches the construct's required class;
7. emit review-level counts, strict counts and binary prevalence flags.
"""
import re
import sys
import unicodedata
from bisect import bisect_right

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/hosp/scripts")
from lexicon import (CONSTRUCTS, AMBIGUOUS, PROVIDER_ANCHORS,  # noqa: E402
                     DESTINATION_ANCHORS, DEST_HOSP_MARKERS,
                     NEGATORS, INTENSIFIERS)

RAW = "/home/claude/reviews_raw.csv"
OUT = "/home/claude/hosp/out/reviews_coded.parquet"

NEG_WINDOW = 3          # tokens scanned to the left for a negator
ANCHOR_WINDOW = 12      # tokens scanned either side for a referent anchor
NEG_MAX_BOUND = 0       # a negator may not cross a clause boundary
ANCHOR_MAX_BOUND = 1    # an anchor may cross at most one clause boundary
ADVERSATIVES = {"mas", "porem", "contudo", "entretanto", "todavia",
                "embora", "apesar"}


# ---------------------------------------------------------------- normalise
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


BOUND = "|"          # strong clause boundary (. ; : ! ? and brackets)
SOFT = ","           # weak clause boundary (comma)


def normalise(s: str) -> str:
    """Casefold, strip diacritics, and turn clause punctuation into an
    explicit boundary token so that negation and referent windows cannot
    silently cross a clause."""
    s = strip_accents(str(s).lower())
    s = re.sub(r"[\-/']+", " ", s)          # keep hyphenated words joinable
    s = re.sub(r"[.;:!?()\[\]\n\r]+", " | ", s)   # strong clause boundary
    s = re.sub(r",+", " , ", s)                     # weak clause boundary
    s = re.sub(r"[^a-z0-9\s|,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------------------------------------------------------------ regex builder
def build(term: str) -> str:
    t = strip_accents(term.lower()).replace("~", r"\w*")
    return r"(?<!\w)" + t + r"(?!\w)"


def compile_group(terms):
    return re.compile("|".join("(?:%s)" % build(t) for t in terms))


COMPILED = {c: {sub: compile_group(t) for sub, t in subs.items()}
            for c, subs in CONSTRUCTS.items()}
COMPILED_AMB = {c: ({sub: compile_group(t) for sub, t in groups.items()}, need)
                for c, (groups, need) in AMBIGUOUS.items()}
RE_PROV = compile_group(PROVIDER_ANCHORS)
RE_DEST = compile_group(DESTINATION_ANCHORS)
RE_DHOSP = re.compile("|".join("(?:%s)" % build(t) for t in DEST_HOSP_MARKERS))
NEG_SET = {strip_accents(n) for n in NEGATORS}
INT_SET = {strip_accents(i) for i in INTENSIFIERS}

# ------------------------------------------------------------- field layout
SUF = ["", "__aff", "__neg", "__prov", "__dest", "__int"]
FIELDS = []
for _c, _subs in CONSTRUCTS.items():
    for _sub in _subs:
        FIELDS += [f"{_c}__{_sub}{s}" for s in SUF]
for _c, (_groups, _need) in AMBIGUOUS.items():
    for _sub in _groups:
        FIELDS += [f"AMB_{_c}__{_sub}", f"AMB_{_c}__{_sub}__ok",
                   f"AMB_{_c}__{_sub}__ok_aff"]
FIELDS += ["DEST_HOSP_marker", "n_tokens"]
FIDX = {f: i for i, f in enumerate(FIELDS)}


# ------------------------------------------------------------------ scoring
def score_review(text):
    out = [0] * len(FIELDS)
    if not text:
        return out

    toks, starts = [], []
    for m in re.finditer(r"\S+", text):
        toks.append(m.group(0))
        starts.append(m.start())
    n = len(toks)
    out[FIDX["n_tokens"]] = n
    if n == 0:
        return out

    is_neg = [t in NEG_SET for t in toks]
    is_int = [t in INT_SET for t in toks]
    # strong boundaries block referent attribution; weak boundaries (commas)
    # only block negation scope
    is_bound = [(t == BOUND or t in ADVERSATIVES) for t in toks]
    is_soft = [(t == SOFT or is_bound[j]) for j, t in enumerate(toks)]
    bcum = [0] * (n + 1)
    for j in range(n):
        bcum[j + 1] = bcum[j] + (1 if is_bound[j] else 0)

    def crossings(i, j):
        lo, hi = (i, j) if i <= j else (j, i)
        return bcum[hi] - bcum[lo + 1] if hi > lo else 0

    def anchor_idx(rx):
        idx = set()
        for m in rx.finditer(text):
            j = bisect_right(starts, m.start()) - 1
            if j >= 0:
                idx.add(j)
        return sorted(idx)

    prov_idx = anchor_idx(RE_PROV)
    dest_idx = anchor_idx(RE_DEST)

    def nearest(sorted_idx, i):
        """Distance to the closest anchor that is inside the token window
        and separated from the hit by at most ANCHOR_MAX_BOUND clauses."""
        if not sorted_idx:
            return None
        k = bisect_right(sorted_idx, i)
        best = None
        for kk in range(max(0, k - 3), min(len(sorted_idx), k + 3)):
            j = sorted_idx[kk]
            d = abs(j - i)
            if d > ANCHOR_WINDOW:
                continue
            if crossings(i, j) > ANCHOR_MAX_BOUND:
                continue
            if best is None or d < best:
                best = d
        return best

    def referent(i):
        dp, dd = nearest(prov_idx, i), nearest(dest_idx, i)
        if dp is None and dd is None:
            return "none"
        if dd is None:
            return "provider"
        if dp is None:
            return "destination"
        return "provider" if dp <= dd else "destination"

    def tok_index(pos):
        i = bisect_right(starts, pos) - 1
        return i if i >= 0 else 0

    def is_negated(i):
        """Scan left for a negator, stopping at the first clause boundary."""
        for j in range(i - 1, max(-1, i - 1 - NEG_WINDOW), -1):
            if is_soft[j]:
                break
            if is_neg[j]:
                return True
        return False

    # -- unambiguous constructs -------------------------------------------
    for cons, subs in COMPILED.items():
        for sub, rx in subs.items():
            b0 = FIDX[f"{cons}__{sub}"]
            for m in rx.finditer(text):
                i = tok_index(m.start())
                out[b0] += 1
                if is_negated(i):
                    out[b0 + 2] += 1
                else:
                    out[b0 + 1] += 1
                if any(is_int[j] and not is_soft[j]
                       for j in range(max(0, i - 2), i)):
                    out[b0 + 5] += 1
                ref = referent(i)
                if ref == "provider":
                    out[b0 + 3] += 1
                elif ref == "destination":
                    out[b0 + 4] += 1

    # -- ambiguous markers: require a matching referent --------------------
    for cons, (groups, need) in COMPILED_AMB.items():
        for sub, rx in groups.items():
            b0 = FIDX[f"AMB_{cons}__{sub}"]
            for m in rx.finditer(text):
                i = tok_index(m.start())
                out[b0] += 1
                if referent(i) == need:
                    out[b0 + 1] += 1
                    if not is_negated(i):
                        out[b0 + 2] += 1

    out[FIDX["DEST_HOSP_marker"]] = len(RE_DHOSP.findall(text))
    return out


# --------------------------------------------------------------------- main
def main():
    df = pd.read_csv(RAW)
    df["review_id"] = np.arange(len(df))
    df["translated"] = (df["lang"] == "pt-BR").astype(int)
    df["pt_corpus"] = df["lang"].isin(["pt", "pt-BR"]).astype(int)

    work = df[df.pt_corpus == 1].copy()
    print(f"corpus for text coding: {len(work):,} reviews "
          f"({len(work)/len(df):.1%} of all rows)", flush=True)

    norm = work["text"].map(normalise)
    from multiprocessing import Pool, cpu_count
    with Pool(max(1, cpu_count() - 1)) as pool:
        rows = pool.map(score_review, norm.tolist(), chunksize=400)
    coded = pd.DataFrame(rows, columns=FIELDS, index=work.index)

    out = pd.concat([work.drop(columns=["text"]), coded], axis=1)
    out["text_norm_len"] = norm.str.len().values

    def s(prefix, suffix):
        cols = [c for c in coded.columns
                if c.startswith(prefix) and c.endswith(suffix)]
        return coded[cols].sum(axis=1) if cols else 0

    # ---- construct totals (strict = unambiguous + anchored ambiguous) ----
    out["HOSP_n"] = s("HOSP__", "__aff")
    out["HOSP_negated_n"] = s("HOSP__", "__neg")
    out["HOSP_prov_n"] = s("HOSP__", "__prov")

    out["SQpos_n"] = s("SQ_POS__", "__aff") + s("AMB_SQ_POS__", "__ok_aff")
    out["SQneg_n"] = (s("SQ_NEG__", "__aff") + s("SQ_POS__", "__neg")
                      + s("AMB_SQ_NEG__", "__ok_aff"))
    out["DQpos_n"] = s("DQ_POS__", "__aff") + s("AMB_DQ_POS__", "__ok_aff")
    out["DQneg_n"] = (s("DQ_NEG__", "__aff") + s("DQ_POS__", "__neg")
                      + s("AMB_DQ_NEG__", "__ok_aff"))
    out["EXP_n"] = s("EXP__", "__aff")
    out["INTpos_n"] = s("INT__", "__aff")
    out["INTneg_n"] = s("INT_NEG__", "__aff")
    out["DESTHOSP_n"] = coded["DEST_HOSP_marker"]

    # permissive variants for sensitivity analysis
    out["SQpos_perm_n"] = s("SQ_POS__", "__aff") + s("AMB_SQ_POS__", "")
    out["DQpos_perm_n"] = s("DQ_POS__", "__aff") + s("AMB_DQ_POS__", "")
    out["DQneg_perm_n"] = s("DQ_NEG__", "__aff") + s("AMB_DQ_NEG__", "")

    for b in ["HOSP", "SQpos", "SQneg", "DQpos", "DQneg", "EXP",
              "INTpos", "INTneg", "DESTHOSP"]:
        out[f"{b}_flag"] = (out[f"{b}_n"] > 0).astype(int)

    for sub in CONSTRUCTS["HOSP"]:
        out[f"HOSP_{sub}_flag"] = (coded[f"HOSP__{sub}__aff"] > 0).astype(int)
    for sub in CONSTRUCTS["DQ_POS"]:
        out[f"DQ_{sub}_flag"] = (coded[f"DQ_POS__{sub}__aff"] > 0).astype(int)
    for sub in CONSTRUCTS["SQ_POS"]:
        out[f"SQ_{sub}_flag"] = (coded[f"SQ_POS__{sub}__aff"] > 0).astype(int)

    out.to_parquet(OUT, index=False)
    print("saved", OUT, out.shape)
    cols = [f"{b}_flag" for b in ["HOSP", "SQpos", "SQneg", "DQpos",
                                  "DQneg", "EXP", "INTpos", "INTneg",
                                  "DESTHOSP"]]
    print(out[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
