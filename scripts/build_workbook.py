# -*- coding: utf-8 -*-
"""
Build the supplementary workbook.

Design rule: wherever the manuscript reports a derived quantity, the workbook
carries the RAW COUNTS and derives the quantity with a live formula, so a
reviewer can see the arithmetic rather than take it on trust. Percentages are
stored as fractions and displayed with a percentage format.
"""
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SUPP = "/home/claude/hosp/out/supp"
DEST = "/mnt/user-data/outputs/Hospitality_Supplementary_Data.xlsx"

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
SUB_FILL = PatternFill("solid", fgColor="EDF2F9")
NOTE_FILL = PatternFill("solid", fgColor="FFF9E6")
HEAD_FONT = Font(name=FONT, size=10, bold=True, color="FFFFFF")
BOLD = Font(name=FONT, size=10, bold=True)
BASE = Font(name=FONT, size=10)
NOTE = Font(name=FONT, size=9, italic=True, color="555555")
TITLE = Font(name=FONT, size=13, bold=True, color="1F3864")
CALC = Font(name=FONT, size=10, color="1F3864")   # blue = live formula
THIN = Side(style="thin", color="BFBFBF")

wb = Workbook()
wb.remove(wb.active)


def sheet(name):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    return ws


def put_title(ws, title, subtitle, row=1):
    ws.cell(row=row, column=1, value=title).font = TITLE
    ws.cell(row=row + 1, column=1, value=subtitle).font = NOTE
    return row + 3


def put_header(ws, row, headers, widths):
    for j, (h, w) in enumerate(zip(headers, widths), start=1):
        c = ws.cell(row=row, column=j, value=h)
        c.font = HEAD_FONT
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border = Border(bottom=THIN)
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[row].height = 30
    ws.freeze_panes = ws.cell(row=row + 1, column=1)
    return row + 1


def put_note(ws, row, text, span=8):
    ws.cell(row=row, column=1, value=text).font = NOTE
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True,
                                                     vertical="top")
    ws.row_dimensions[row].height = 30
    return row + 2


def w(ws, r, c, v, fmt=None, font=None, bold=False):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font = font or (BOLD if bold else BASE)
    if fmt:
        cell.number_format = fmt
    return cell


PCT = "0.0%"
PCT2 = "0.00%"
NUM = "#,##0"
DEC3 = "0.000"
DEC4 = "0.0000"

# =========================================================== S1 contents
ws = sheet("S1. Contents")
r = put_title(ws, "Supplementary data",
              "Destination Hospitality as an Emergent Property of Commercial "
              "Host–Guest Encounters: Evidence from 50,180 Visitor Reviews "
              "across Twenty-Nine Emerging-Economy Cities")
ws.cell(row=r, column=1, value="Every derived quantity in this workbook is a "
        "live formula over the raw counts in the same sheet. "
        "Figures shown in blue are computed by the spreadsheet; black figures "
        "are inputs from the analysis pipeline. "
        "Percentages are stored as fractions.").font = NOTE
ws.merge_cells(start_row=r, start_column=1, end_row=r + 1, end_column=6)
ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical="top")
r += 3

r = put_header(ws, r, ["Sheet", "Contents", "Corresponds to"],
               [30, 74, 34])
contents = [
    ("S2. Corpus", "Reviews, suppliers, mean rating, translation share and construct counts for each of the 29 destinations", "Table 1; Section 3.1"),  # noqa: E501
    ("S3. Prevalence", "Review-level prevalence of every coded construct "
     "across the analytic corpus", "Sections 4.1–4.2"),
    ("S4. Marker set", "The 56 induced hospitality marker groups: surface forms, assigned type, prevalence, contrastive scores", "Section 3.3; Table 2"),  # noqa: E501
    ("S5. City index", "Destination hospitality index: raw counts, bootstrap "
     "intervals, and the three adjusted variants",
     "Figure 1; Section 4.2.2"),
    ("S6. City profile", "Composition of hospitality by type within each city, among hospitality-bearing reviews",  # noqa: E501
     "Figure 2; Section 4.2.3"),
    ("S7. Reliability", "Split-half correlations, Spearman–Brown reliability, "
     "permutation tests and ICC(1)", "Table 4"),
    ("S8. Rating model", "Convergent validity: star rating on construct "
     "mentions, supplier fixed effects", "Table 5"),
    ("S9. RQ2 models", "Whether the commercial encounter reaches the "
     "destination: Models A, A2 and B", "Table 6; Figure 3"),
    ("S10. Leave-one-out", "RQ2 coefficients and city-level correlations with "
     "each country omitted in turn", "Section 4.3"),
    ("S11. Correlations",
     "Spearman correlation matrix of the city-level construct measures", "Section 4.2.3"),  # noqa: E501
    ("S12. Robustness", "Rank correlations of the index under each robustness "
     "condition",
     "Table 7; Section 4.4"),
    ("S13. Attribution",
     "How often reviews name a specific individual, by construct", "Section 4.2.3"),  # noqa: E501
    ("S14. Induced terms", "The full list of 354 discriminant hospitality "
     "terms with their contrastive scores", "Section 3.3"),
    ("S15. Co-occurrence", "Pairwise co-occurrence and phi coefficients "
     "between the three hospitality types", "Section 4.1"),
    ("S16. Induction audit", "The vocabulary induction funnel, the threshold sensitivity sweep, the borderline terms, and the k-selection table", "Sections 3.3–3.4"),  # noqa: E501
]
for name, desc, corr in contents:
    w(ws, r, 1, name, bold=True)
    w(ws, r, 2, desc)
    ws.cell(row=r, column=2).alignment = Alignment(
        wrap_text=True, vertical="top")
    w(ws, r, 3, corr)
    ws.row_dimensions[r].height = 26
    r += 1

r += 1
w(ws, r, 1, "Provenance", bold=True)
r += 1
prov = [
    ("Source data", "reviews_raw.csv — 54,687 Google Maps reviews of 11,477 "
     "tourism suppliers, 29 cities, 10 emerging economies"),
    ("Analytic corpus", "50,180 reviews (91.8%). Excluded: 3,559 with a star "
     "rating but no text; 948 served in a language other than Portuguese"),
    ("Language", "74.5% of the text is machine-translated into Brazilian "
     "Portuguese by the platform; 25.5% is original Portuguese"),
    ("Segments", "Mass-market and independent/creative suppliers are pooled; "
     "segment is retained as a covariate and as a robustness split"),
    ("Random seeds", "Skip-gram model seed 42. Every bootstrap, permutation and split-half draw is seeded from the name of the statistic being computed, so no result depends on the order in which the scripts consume the generator"),  # noqa: E501
    ("Software", "Python 3.11, gensim 4.4, scikit-learn 1.8, statsmodels 0.15, pandas 3.0"),  # noqa: E501
]
for k, v in prov:
    w(ws, r, 1, k, bold=True)
    w(ws, r, 2, v)
    ws.cell(row=r, column=2).alignment = Alignment(
        wrap_text=True, vertical="top")
    ws.row_dimensions[r].height = 26
    r += 1

# =========================================================== S2 corpus
d2 = pd.read_csv(f"{SUPP}/S2_corpus.csv").sort_values(["country", "city"])
ws = sheet("S2. Corpus")
r = put_title(ws, "S2. Corpus by destination",
              "Counts are inputs; every share and mean is a live formula. "
              "Corresponds to Table 1 of the manuscript.")
hdr = ["City", "Country", "Suppliers", "Reviews", "Sum of ratings",
       "Mean rating", "Translated (n)", "Translated (%)",
       "Creative segment (n)", "Creative (%)", "Sum of tokens",
       "Mean tokens", "Hospitality (n)", "Hospitality (%)",
       "Type 1 conduct (n)", "Type 1 (%)", "Type 2 ambience (n)", "Type 2 (%)",
       "Type 3 development (n)", "Type 3 (%)",
       "Service quality + (n)", "SQ + (%)", "Service quality − (n)", "SQ − (%)",  # noqa: E501
       "Destination quality + (n)", "DQ + (%)",
       "Destination quality − (n)", "DQ − (%)",
       "Generic affect (n)", "Affect (%)",
       "Names a person (n)", "Names a person (%)"]
widths = [16, 17] + [10, 9, 11, 9, 11, 10, 12, 9, 11, 10] + [11, 10] * 10
r0 = put_header(ws, r, hdr, widths)

start = r0
for i, row in enumerate(d2.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.city)
    w(ws, rr, 2, row.country)
    w(ws, rr, 3, row.suppliers, NUM)
    w(ws, rr, 4, row.reviews, NUM)
    w(ws, rr, 5, row.rating_sum, NUM)
    w(ws, rr, 6, f"=E{rr}/D{rr}", "0.000", font=CALC)
    w(ws, rr, 7, row.translated_n, NUM)
    w(ws, rr, 8, f"=G{rr}/D{rr}", PCT, font=CALC)
    w(ws, rr, 9, row.creative_n, NUM)
    w(ws, rr, 10, f"=I{rr}/D{rr}", PCT, font=CALC)
    w(ws, rr, 11, row.tokens_sum, NUM)
    w(ws, rr, 12, f"=K{rr}/D{rr}", "0.0", font=CALC)
    pairs = [(13, row.H_n), (15, row.H1_n), (17, row.H2_n), (19, row.H3_n),
             (21, row.SQpos_n), (23, row.SQneg_n), (25, row.DQpos_n),
             (27, row.DQneg_n), (29, row.EXP_n), (31, row.named_n)]
    for col, val in pairs:
        w(ws, rr, col, int(val), NUM)
        w(ws, rr, col + 1,
          f"={get_column_letter(col)}{rr}/D{rr}", PCT, font=CALC)
end = r0 + len(d2) - 1

tot = end + 1
w(ws, tot, 1, "All destinations", bold=True)
w(ws, tot, 2, "10 countries", bold=True)
for col in [3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
    L = get_column_letter(col)
    w(ws, tot, col, f"=SUM({L}{start}:{L}{end})", NUM, font=CALC, bold=True)
ws.cell(row=tot, column=3).number_format = NUM
w(ws, tot, 6, f"=E{tot}/D{tot}", "0.000", font=CALC, bold=True)
for col in [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
    prev = get_column_letter(col - 1)
    fmt = "0.0" if col == 12 else PCT
    w(ws, tot, col, f"={prev}{tot}/D{tot}", fmt, font=CALC, bold=True)
for j in range(1, 33):
    ws.cell(row=tot, column=j).border = Border(top=THIN)
put_note(ws, tot + 2,
         "Note. Suppliers are counted within each city; the total is the sum "
         "of city-level supplier counts. "
         "'Names a person' counts reviews mentioning a first name from a "
         "gazetteer built from the platform's own author-name "
         "field, excluding the reviewer's own name and any name-like common "
         "word (see S13).", span=12)

# =========================================================== S3 prevalence
d3 = pd.read_csv(f"{SUPP}/S3_prevalence.csv")
ws = sheet("S3. Prevalence")
r = put_title(ws, "S3. Review-level prevalence of every coded construct",
              "Denominator is the analytic corpus of 50,180 reviews.")
r0 = put_header(ws, r, ["Indicator", "Construct", "Reviews flagged",
                        "Corpus size", "Prevalence"],
                [20, 46, 15, 13, 13])
for i, row in enumerate(d3.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.indicator)
    w(ws, rr, 2, row.label)
    w(ws, rr, 3, row.n_reviews_flagged, NUM)
    w(ws, rr, 4, row.n_reviews_total, NUM)
    w(ws, rr, 5, f"=C{rr}/D{rr}", PCT2, font=CALC)
put_note(ws, r0 + len(d3) + 1,
         "Note. 'Hospitality predicated of the destination itself' counts "
         "reviews that call a city, a country or its residents "
         "hospitable, as opposed to a commercial host. Its near-zero "
         "prevalence is the result reported in Section 4.2.1.", span=5)

# =========================================================== S4 markers
d4 = pd.read_csv(f"{SUPP}/S4_markers.csv")
ws = sheet("S4. Marker set")
r = put_title(ws, "S4. The 56 induced hospitality marker groups",
              "Induced by contrastive seed expansion (Section 3.3) and "
              "assigned "
              "to a type by k-means on the embedding space (Section 3.4).")
r0 = put_header(ws, r, ["Marker group (surface forms matched)", "Stem label",
                        "Type", "Reviews", "Corpus", "Prevalence",
                        "Similarity to hospitality centroid",
                        "Contrastive margin over generic praise",
                        "Centrality within type"],
                [52, 11, 24, 11, 10, 12, 15, 16, 14])
for i, row in enumerate(d4.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.family)
    w(ws, rr, 2, row.group)
    w(ws, rr, 3, row.type_label)
    w(ws, rr, 4, row.n_reviews, NUM)
    w(ws, rr, 5, row.n_total, NUM)
    w(ws, rr, 6, f"=D{rr}/E{rr}", PCT2, font=CALC)
    w(ws, rr, 7, row.sim_hosp, DEC3)
    w(ws, rr, 8, row.margin, DEC3)
    w(ws, rr, 9, row.centrality, DEC3)
put_note(ws, r0 + len(d4) + 1,
         "Note. A marker group is a set of agreement variants of one lexical "
         "item. Occurrence is matched against the explicit "
         "surface forms listed in column A, never against a prefix. The "
         "contrastive margin is the term's cosine similarity to the "
         "hospitality seed centroid minus its similarity to a centroid of ten "
         "generic praise terms; only terms with a positive "
         "margin are retained.", span=9)

# =========================================================== S5 city index
d5 = pd.read_csv(f"{SUPP}/S5_city_index.csv")
d5["prevalence"] = d5.n_hosp / d5.n_reviews
d5 = d5.sort_values("prevalence", ascending=False)
ws = sheet("S5. City index")
r = put_title(ws, "S5. Destination hospitality index",
              "Corresponds to Figure 1. Bootstrap intervals resample "
              "suppliers "
              "(500 replicates); adjusted indices are city fixed effects on "
              "the logit scale.")
r0 = put_header(ws, r, ["City", "Country", "Suppliers", "Reviews",
                        "Hospitality-bearing reviews", "Prevalence",
                        "Bootstrap 2.5%", "Bootstrap 97.5%", "Bootstrap SE",
                        "Interval width",
                        "Index adjusted for composition",
                        "…also net of generic affect",
                        "…also net of the star rating"],
                [17, 18, 10, 10, 14, 11, 11, 11, 11, 11, 14, 14, 14])
for i, row in enumerate(d5.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.city)
    w(ws, rr, 2, row.country)
    w(ws, rr, 3, row.n_suppliers, NUM)
    w(ws, rr, 4, row.n_reviews, NUM)
    w(ws, rr, 5, row.n_hosp, NUM)
    w(ws, rr, 6, f"=E{rr}/D{rr}", PCT, font=CALC)
    w(ws, rr, 7, row.boot_lo, PCT2)
    w(ws, rr, 8, row.boot_hi, PCT2)
    w(ws, rr, 9, row.boot_se, DEC4)
    w(ws, rr, 10, f"=H{rr}-G{rr}", PCT2, font=CALC)
    w(ws, rr, 11, row.index_adjusted, DEC3)
    w(ws, rr, 12, row.index_net_affect, DEC3)
    w(ws, rr, 13, row.index_net_affect_rating, DEC3)
e5 = r0 + len(d5) - 1
tot = e5 + 1
w(ws, tot, 1, "All destinations", bold=True)
for col in [3, 4, 5]:
    L = get_column_letter(col)
    w(ws, tot, col, f"=SUM({L}{r0}:{L}{e5})", NUM, font=CALC, bold=True)
w(ws, tot, 6, f"=E{tot}/D{tot}", PCT, font=CALC, bold=True)
for j in range(1, 14):
    ws.cell(row=tot, column=j).border = Border(top=THIN)
put_note(ws, tot + 2,
         "Note. The adjusted indices are the city fixed effects of a logistic "
         "model for whether a review carries a hospitality "
         "marker, controlling for segment, log review length, translation "
         "status and recency, centred on the corpus mean. Column L "
         "adds generic experiential affect to that model and column M adds "
         "the star rating as well; a city that keeps a high value "
         "in column M is not simply a city whose reviewers write "
         "enthusiastically.", span=13)

# =========================================================== S6 profile
d6 = pd.read_csv(f"{SUPP}/S6_city_profile.csv")
ws = sheet("S6. City profile")
r = put_title(ws, "S6. Composition of hospitality within each city",
              "Corresponds to Figure 2. Shares are computed among "
              "hospitality-bearing reviews only, so they are independent of "
              "how "
              "expressive the city's review culture is.")
r0 = put_header(ws, r, ["City", "Country", "Hospitality-bearing reviews",
                        "Type 1 conduct (n)", "Type 2 ambience (n)",
                        "Type 3 development (n)", "Sum of type mentions",
                        "Type 1 share", "Type 2 share", "Type 3 share",
                        "Shares sum to"],
                [17, 18, 15, 13, 13, 14, 13, 12, 12, 12, 11])
for i, row in enumerate(d6.sort_values("city").itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.city)
    w(ws, rr, 2, row.country)
    w(ws, rr, 3, row.n_hosp_reviews, NUM)
    w(ws, rr, 4, row.n_conduct, NUM)
    w(ws, rr, 5, row.n_ambience, NUM)
    w(ws, rr, 6, row.n_development, NUM)
    w(ws, rr, 7, f"=SUM(D{rr}:F{rr})", NUM, font=CALC)
    w(ws, rr, 8, f"=D{rr}/$G{rr}", PCT, font=CALC)
    w(ws, rr, 9, f"=E{rr}/$G{rr}", PCT, font=CALC)
    w(ws, rr, 10, f"=F{rr}/$G{rr}", PCT, font=CALC)
    w(ws, rr, 11, f"=SUM(H{rr}:J{rr})", PCT, font=CALC)
put_note(ws, r0 + len(d6) + 1,
         "Note. A review may carry more than one type, so the sum of type "
         "mentions (column G) exceeds the number of "
         "hospitality-bearing reviews (column C). Shares are taken over the "
         "type mentions, which is why column K equals 100% "
         "in every row — it is a check, not a result.", span=11)

# =========================================================== S7 reliability
d7 = pd.read_csv(f"{SUPP}/S7_reliability.csv")
d7b = pd.read_csv(f"{SUPP}/S7b_profile_reliability.csv")
ws = sheet("S7. Reliability")
r = put_title(ws, "S7. Reliability and discrimination of the city-level "
              "measures",
              "Corresponds to Table 4. The Spearman–Brown column is a live "
              "formula over the split-half correlation to its left.")
r0 = put_header(ws, r, ["Indicator", "Construct",
                        "Split-half correlation r",
                        "Spearman–Brown 2r/(1+r)",
                        "r 2.5%", "r 97.5%",
                        "Spearman–Brown 2.5%", "Spearman–Brown 97.5%",
                        "Between-city variance", "Permutation p", "ICC(1)"],
                [22, 40, 13, 14, 10, 10, 13, 13, 14, 12, 10])
rr = r0
for row in d7.itertuples():
    w(ws, rr, 1, row.indicator)
    w(ws, rr, 2, row.label)
    w(ws, rr, 3, row.r_half, DEC3)
    w(ws, rr, 4, f"=2*C{rr}/(1+C{rr})", DEC3, font=CALC)
    w(ws, rr, 5, row.r_half_lo, DEC3)
    w(ws, rr, 6, row.r_half_hi, DEC3)
    w(ws, rr, 7, f"=2*E{rr}/(1+E{rr})", DEC3, font=CALC)
    w(ws, rr, 8, f"=2*F{rr}/(1+F{rr})", DEC3, font=CALC)
    w(ws, rr, 9, row.between_city_var, "0.00000")
    w(ws, rr, 10, row.perm_p, DEC3)
    w(ws, rr, 11, row.icc1, DEC3)
    rr += 1
rr += 1
w(ws, rr, 1, "Within hospitality-bearing reviews (the composition of S6)", bold=True)  # noqa: E501
rr += 1
for row in d7b.itertuples():
    w(ws, rr, 1, row.indicator)
    w(ws, rr, 2, row.label)
    w(ws, rr, 3, row.r_half, DEC3)
    w(ws, rr, 4, f"=2*C{rr}/(1+C{rr})", DEC3, font=CALC)
    w(ws, rr, 5, row.r_half_lo, DEC3)
    w(ws, rr, 6, row.r_half_hi, DEC3)
    w(ws, rr, 7, f"=2*E{rr}/(1+E{rr})", DEC3, font=CALC)
    w(ws, rr, 8, f"=2*F{rr}/(1+F{rr})", DEC3, font=CALC)
    w(ws, rr, 9, row.between_city_var, "0.00000")
    w(ws, rr, 10, row.perm_p, DEC3)
    w(ws, rr, 11, row.icc1, DEC3)
    rr += 1
put_note(ws, rr + 1,
         "Note. Halves are formed by splitting SUPPLIERS within each city, "
         "not reviews, so the two halves are genuinely "
         "independent; 200 replications (150 for the profile block). The "
         "permutation test shuffles city labels across suppliers "
         "(1,000 permutations); p = .001 is the smallest attainable value. "
         "ICC(1) is small throughout because most variance in "
         "whether a review mentions hospitality lies between reviews and "
         "suppliers rather than between cities — which is why the "
         "city means carry bootstrap intervals in S5. Columns G and H "
         "reproduce the interval as reported in Table 4 of the "
         "manuscript, which is on the Spearman-Brown scale.", span=11)

# =========================================================== S8 rating model
d8 = pd.read_csv(f"{SUPP}/S8_rating_model.csv")
LBL = {
    "H_conduct": "Hospitality type 1: interpersonal conduct",
    "H_ambience": "Hospitality type 2: created ambience",
    "H_development": "Hospitality type 3: development",
    "SQpos_flag": "Service quality, positive",
    "SQneg_flag": "Service quality, negative",
    "DQpos_flag": "Destination quality, positive",
    "DQneg_flag": "Destination quality, negative",
    "EXP_flag": "Generic experiential affect",
    "log_len": "Log review length (control)",
    "translated": "Machine-translated (control)",
    "is_creative": "Creative segment (control)",
    "years_ago": "Years since the review (control)",
}
ws = sheet("S8. Rating model")
r = put_title(ws, "S8. Convergent validity: star rating on construct mentions",
              "Corresponds to Table 5. Within estimator with supplier fixed "
              "effects absorbed by demeaning; standard errors clustered on "
              "the supplier.")
r0 = put_header(ws, r, ["Predictor", "Coefficient (stars)", "SE", "t",
                        "p", "95% lower", "95% upper", "Significant at 5%"],
                [42, 15, 11, 10, 12, 11, 11, 14])
for i, row in enumerate(d8.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, LBL.get(row.predictor, row.predictor))
    w(ws, rr, 2, row.coef, DEC3)
    w(ws, rr, 3, row.se, DEC4)
    w(ws, rr, 4, f"=B{rr}/C{rr}", "0.00", font=CALC)
    w(ws, rr, 5, row.p, "0.000E+00")
    w(ws, rr, 6, f"=B{rr}-1.96*C{rr}", DEC3, font=CALC)
    w(ws, rr, 7, f"=B{rr}+1.96*C{rr}", DEC3, font=CALC)
    w(ws, rr, 8, f'=IF(E{rr}<0.05,"yes","no")', font=CALC)
rr = r0 + len(d8) + 1
for k, v in [("Observations", int(d8.n.iloc[0])),
             ("Suppliers (fixed effects absorbed)", int(d8.suppliers.iloc[0])),
             ("Within R²", float(d8.within_r2.iloc[0]))]:
    w(ws, rr, 1, k, bold=True)
    w(ws, rr, 2, v, NUM if isinstance(v, int) else DEC4, bold=True)
    rr += 1
put_note(ws, rr + 1,
         "Note. The outcome is the one-to-five star rating. Because the "
         "supplier effect is absorbed, every coefficient is a "
         "within-supplier comparison: it contrasts reviews of the SAME firm "
         "that do and do not mention the construct. Segment is "
         "absorbed by the supplier effect and its coefficient is not "
         "identified.", span=8)

# =========================================================== S9 RQ2
d9 = pd.read_csv(f"{SUPP}/S9_rq2_modelA.csv")
d9b = pd.read_csv(f"{SUPP}/S9b_rq2_modelB.csv")
ws = sheet("S9. RQ2 models")
r = put_title(ws, "S9. Does the commercial encounter reach the destination?",
              "Corresponds to Table 6 and Figure 3. Model A is the headline "
              "specification; A2 replaces supplier with city fixed effects; "
              "Model B conditions on the destination being mentioned.")
for model, label in [("A: supplier fixed effects",
                      "Model A — linear probability that the review "
                      "evaluates the destination (supplier fixed effects)"),
                     ("A2: city fixed effects",
                      "Model A2 — same model with city fixed effects instead "
                      "of supplier fixed effects")]:
    w(ws, r, 1, label, bold=True)
    r += 1
    r0 = put_header(ws, r, ["Predictor", "Coefficient", "SE", "t", "p",
                            "95% lower", "95% upper", "Significant at 5%"],
                    [42, 13, 11, 10, 12, 11, 11, 14])
    sub = d9[d9.model == model]
    for i, row in enumerate(sub.itertuples()):
        rr = r0 + i
        w(ws, rr, 1, LBL.get(row.predictor, row.predictor))
        w(ws, rr, 2, row.coef, DEC4)
        w(ws, rr, 3, row.se, DEC4)
        w(ws, rr, 4, f"=B{rr}/C{rr}", "0.00", font=CALC)
        w(ws, rr, 5, row.p, "0.000E+00")
        w(ws, rr, 6, f"=B{rr}-1.96*C{rr}", DEC4, font=CALC)
        w(ws, rr, 7, f"=B{rr}+1.96*C{rr}", DEC4, font=CALC)
        w(ws, rr, 8, f'=IF(E{rr}<0.05,"yes","no")', font=CALC)
    r = r0 + len(sub) + 2

w(ws, r, 1, "Model B — logistic model of destination valence, among the "
  "reviews that evaluate the destination", bold=True)
r += 1
r0 = put_header(ws, r, ["Predictor", "Coefficient (log-odds)", "SE",
                        "Odds ratio", "p", "OR 95% lower", "OR 95% upper",
                        "Significant at 5%"],
                [42, 15, 11, 12, 12, 12, 12, 14])
for i, row in enumerate(d9b.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, LBL.get(row.predictor, row.predictor))
    w(ws, rr, 2, row.coef, DEC3)
    w(ws, rr, 3, row.se, DEC4)
    w(ws, rr, 4, f"=EXP(B{rr})", DEC3, font=CALC)
    w(ws, rr, 5, row.p, "0.000E+00")
    w(ws, rr, 6, f"=EXP(B{rr}-1.96*C{rr})", DEC3, font=CALC)
    w(ws, rr, 7, f"=EXP(B{rr}+1.96*C{rr})", DEC3, font=CALC)
    w(ws, rr, 8, f'=IF(E{rr}<0.05,"yes","no")', font=CALC)
rr = r0 + len(d9b) + 1
w(ws, rr, 1, "Observations (reviews evaluating the destination)", bold=True)
w(ws, rr, 2, int(d9b.n.iloc[0]), NUM, bold=True)
put_note(ws, rr + 2,
         "Note. The headline result is the first row of Model A: positive "
         "service quality has no association with whether a review "
         "evaluates the destination, while all three hospitality types do. "
         "Model A2 shows the result is not an artefact of the "
         "within-supplier design — without supplier fixed effects the "
         "service-quality coefficient moves further below zero, the "
         "direction opposite to a halo. Model B is estimated on the reviews "
         "that mention the destination, so it describes valence "
         "rather than salience.", span=8)

# =========================================================== S10 LOO
d10 = pd.read_csv(f"{SUPP}/S10_loo.csv")
ws = sheet("S10. Leave-one-out")
r = put_title(ws, "S10. Leave-one-country-out analysis of RQ2",
              "Each row re-estimates Model A and the city-level correlation "
              "with one country dropped, testing whether the null result "
              "depends on any single national sample.")
r0 = put_header(ws, r, ["Country omitted", "Reviews retained",
                        "Service quality + coefficient", "SE", "p",
                        "Service quality − coefficient",
                        "Type 1 conduct", "Type 2 ambience",
                        "Type 3 development",
                        "City-level rho (SQ ~ DQ)", "p", "Cities"],
                [19, 13, 15, 10, 10, 15, 12, 12, 13, 14, 10, 9])
for i, row in enumerate(d10.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.omitted_country)
    w(ws, rr, 2, row.n_reviews, NUM)
    w(ws, rr, 3, row.SQpos_coef, DEC4)
    w(ws, rr, 4, row.SQpos_se, DEC4)
    w(ws, rr, 5, row.SQpos_p, DEC3)
    w(ws, rr, 6, row.SQneg_coef, DEC4)
    w(ws, rr, 7, row.H_conduct_coef, DEC4)
    w(ws, rr, 8, row.H_ambience_coef, DEC4)
    w(ws, rr, 9, row.H_development_coef, DEC4)
    w(ws, rr, 10, row.city_rho_SQ_DQ, DEC3)
    w(ws, rr, 11, row.city_rho_p, DEC3)
    w(ws, rr, 12, row.n_cities, NUM)
e10 = r0 + len(d10) - 1
rr = e10 + 2
for lab, col in [("Minimum", "MIN"), ("Maximum", "MAX")]:
    w(ws, rr, 1, lab, bold=True)
    for c in [3, 5, 7, 8, 9, 10]:
        L = get_column_letter(c)
        w(ws, rr, c, f"={col}({L}{r0}:{L}{e10})",
          DEC3 if c in (5, 10) else DEC4, font=CALC, bold=True)
    rr += 1
put_note(ws, rr + 1,
         "Note. Across all ten omissions the service-quality coefficient "
         "never departs from zero by more than 0.003 and its "
         "p-value never falls below .48, while the three hospitality "
         "coefficients stay positive and significant. The city-level "
         "correlation between service quality and destination quality is "
         "non-significant under every omission.", span=12)

# =========================================================== S11 correlations
d11 = pd.read_csv(f"{SUPP}/S11_city_correlations.csv", index_col=0)
ws = sheet("S11. Correlations")
r = put_title(ws, "S11. Spearman correlations among the city-level measures",
              "n = 29 cities. The high correlations in the last two columns "
              "are "
              "the discriminant-validity exposure discussed in Section 4.2.3.")
r0 = put_header(ws, r, [""] + list(d11.columns),
                [26] + [14] * len(d11.columns))
for i, (name, row) in enumerate(d11.iterrows()):
    rr = r0 + i
    w(ws, rr, 1, name, bold=True)
    for j, v in enumerate(row.values, start=2):
        c = w(ws, rr, j, float(v), DEC3)
        if abs(v) > 0.79 and abs(v) < 0.999:
            c.fill = NOTE_FILL
put_note(ws, r0 + len(d11) + 1,
         "Note. Shaded cells are correlations above .79 in absolute value. "
         "The raw hospitality index correlates .92 with generic "
         "experiential affect and .81 with the share of five-star ratings; "
         "this is why the manuscript relies on the adjusted index "
         "of S5 (columns K–M) and on the composition of S6, which is "
         "arithmetically immune to expressiveness.",
         span=len(d11.columns) + 1)

# =========================================================== S12 robustness
d12 = pd.read_csv(f"{SUPP}/S12_robustness.csv")
ws = sheet("S12. Robustness")
r = put_title(ws, "S12. Robustness of the city index",
              "Corresponds to Table 7. Spearman rank correlations against the "
              "main index across the twenty-nine cities.")
r0 = put_header(ws, r, ["Test", "What it addresses", "Reviews", "Cities",
                        "Rank correlation with the main index"],
                [36, 52, 11, 9, 16])
for i, row in enumerate(d12.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.test)
    w(ws, rr, 2, row.addresses)
    ws.cell(row=rr, column=2).alignment = Alignment(
        wrap_text=True, vertical="top")
    w(ws, rr, 3, row.n_reviews, NUM)
    w(ws, rr, 4, row.n_cities, NUM)
    w(ws, rr, 5, row.rank_rho, DEC3)
    ws.row_dimensions[rr].height = 26
put_note(ws, r0 + len(d12) + 1,
         "Note. The translated-only test bounds rather than resolves the "
         "translation confound: the Brazilian cities keep only "
         "their small, self-selected translated subsets (42 to 565 reviews). "
         "The second row compares the composition-adjusted "
         "index with the index that additionally removes generic affect and "
         "the star rating.", span=5)

# =========================================================== S13 attribution
d13 = pd.read_csv(f"{SUPP}/S13_attribution.csv")
ws = sheet("S13. Attribution")
r = put_title(ws, "S13. How often a review names a specific individual",
              "Supports the reading that hospitality markers capture "
              "encounters "
              "rather than sentiment (Section 4.2.3).")
r0 = put_header(ws, r, ["Subset of reviews", "Reviews",
                        "Naming a person other than the reviewer",
                        "Share"], [34, 13, 20, 12])
for i, row in enumerate(d13.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.subset)
    w(ws, rr, 2, row.n_reviews, NUM)
    w(ws, rr, 3, row.n_naming_a_person, NUM)
    w(ws, rr, 4, f"=C{rr}/B{rr}", PCT, font=CALC)
rr = r0 + len(d13) + 1
w(ws, rr, 1, "Difference, hospitality-bearing minus no hospitality marker", bold=True)  # noqa: E501
w(ws, rr, 4, f"=D{r0+1}-D{r0+2}", PCT, font=CALC, bold=True)
put_note(ws, rr + 2,
         "Note. The gazetteer retains only tokens that are common as author "
         "names in the corpus and not markedly more frequent as "
         "ordinary words, so words such as 'rosa' and 'gloria' are excluded. "
         "The measure is therefore a lower bound. "
         "Chi-square = 1,189.3, p < .001, Cramér's V = .154 for the "
         "hospitality contrast.", span=4)

# =========================================================== S14 terms
d14 = pd.read_csv(f"{SUPP}/S14_terms.csv")
ws = sheet("S14. Induced terms")
r = put_title(ws, "S14. The full induced hospitality vocabulary",
              "All 354 terms retained by the contrastive filter, before "
              "grouping "
              "into the 56 marker groups of S4 and before the prevalence "
              "threshold.")
r0 = put_header(ws, r, ["Term", "Similarity to hospitality centroid",
                        "Similarity to generic-praise centroid",
                        "Contrastive margin", "Corpus frequency",
                        "Marker group"],
                [24, 16, 17, 14, 14, 13])
for i, row in enumerate(d14.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.term)
    w(ws, rr, 2, row.sim_hosp, DEC3)
    w(ws, rr, 3, row.sim_praise, DEC3)
    w(ws, rr, 4, f"=B{rr}-C{rr}", DEC3, font=CALC)
    w(ws, rr, 5, row.freq, NUM)
    w(ws, rr, 6, row.group)
put_note(ws, r0 + len(d14) + 1,
         "Note. Retention required a similarity of at least .42 to the "
         "hospitality centroid AND a contrastive margin of at least "
         ".02 over generic praise. Personal names and topical terms were "
         "removed beforehand. Terms whose marker group fell below "
         "the 0.3% prevalence threshold are listed here but do not enter the "
         "analysis.", span=6)

# =========================================================== S15 co-occurrence
d15 = pd.read_csv(f"{SUPP}/S15_cooccurrence.csv")
ws = sheet("S15. Co-occurrence")
r = put_title(ws, "S15. Co-occurrence of the three hospitality types",
              "Evidence that the types are distinguishable rather than facets "
              "of "
              "a single enthusiasm (Section 4.1).")
r0 = put_header(ws, r, ["Type A", "Type B", "Reviews with A", "Reviews with B",
                        "Reviews with both", "Corpus",
                        "Expected if independent", "Observed / expected",
                        "Phi coefficient"],
                [38, 38, 12, 12, 13, 10, 14, 13, 12])
for i, row in enumerate(d15.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.type_a)
    w(ws, rr, 2, row.type_b)
    w(ws, rr, 3, row.n_a, NUM)
    w(ws, rr, 4, row.n_b, NUM)
    w(ws, rr, 5, row.n_both, NUM)
    w(ws, rr, 6, row.n_total, NUM)
    w(ws, rr, 7, f"=C{rr}*D{rr}/F{rr}", "#,##0.0", font=CALC)
    w(ws, rr, 8, f"=E{rr}/G{rr}", "0.00", font=CALC)
    w(ws, rr, 9, row.phi, DEC3)
put_note(ws, r0 + len(d15) + 1,
         "Note. The types co-occur more than chance would predict — they are "
         "all hospitality — but the phi coefficients of .21 to "
         ".23 are far from the unity that would indicate a single underlying "
         "dimension. 57.6% of reviews carry none of the three.",
         span=9)

# =========================================================== S16 induction
f16 = pd.read_csv(f"{SUPP}/S16_funnel.csv")
s16 = pd.read_csv(f"{SUPP}/S16_sensitivity.csv")
m16 = pd.read_csv(f"{SUPP}/S16_margin_cases.csv")
k16 = pd.read_csv(f"{SUPP}/S16_k_selection.csv")

ws = sheet("S16. Induction audit")
r = put_title(ws, "S16. Auditing the vocabulary induction",
              "How the 6,235-term corpus vocabulary becomes the 56-marker "
              "instrument, how much the retention thresholds matter, and how "
              "the number of types was chosen. Corresponds to Sections "
              "3.3–3.4.")

w(ws, r, 1, "Panel A — the induction funnel at the reported thresholds "
  "(similarity ≥ .42, contrastive margin ≥ .02, prevalence ≥ 0.3%)", bold=True)
r += 1
r0 = put_header(ws, r, ["Stage", "Terms surviving", "Removed at this stage",
                        "Share of the corpus vocabulary"],
                [58, 15, 17, 18])
for i, row in enumerate(f16.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.stage)
    w(ws, rr, 2, row.n, NUM)
    if i == 0:
        w(ws, rr, 3, "—")
    else:
        w(ws, rr, 3, f"=B{rr-1}-B{rr}", NUM, font=CALC)
    w(ws, rr, 4, f"=B{rr}/$B${r0}", PCT2, font=CALC)
r = r0 + len(f16) + 1
r = put_note(ws, r,
             "Note. The last two rows count marker groups rather than terms, "
             "so the 'removed' column changes meaning there: "
             "234 groups are formed from 354 surface forms, and 178 of those "
             "groups fall below the prevalence threshold. "
             "The single largest discretionary step is the removal of 146 "
             "topical terms, described in Section 3.3.", span=4)

w(ws, r, 1, "Panel B — threshold sensitivity: the instrument rebuilt end to "
  "end "
  "under seven settings", bold=True)
r += 1
r0 = put_header(ws, r, ["Similarity threshold", "Margin threshold",
                        "Prevalence threshold", "Terms retained",
                        "Marker groups", "Corpus prevalence",
                        "Rank correlation with the reported index", "Setting"],
                [15, 14, 15, 13, 13, 14, 18, 12])
for i, row in enumerate(s16.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.sim_threshold, "0.00")
    w(ws, rr, 2, row.margin_threshold, "0.00")
    w(ws, rr, 3, row.prevalence_threshold, "0.000")
    w(ws, rr, 4, row.n_terms, NUM)
    w(ws, rr, 5, row.n_marker_groups, NUM)
    w(ws, rr, 6, row.corpus_prevalence, PCT)
    w(ws, rr, 7, row.rank_rho_vs_reported, DEC3)
    lab = row.setting if isinstance(row.setting, str) else ""
    c = w(ws, rr, 8, lab, bold=bool(lab))
    if lab:
        for j in range(1, 9):
            ws.cell(row=rr, column=j).fill = NOTE_FILL
r = r0 + len(s16) + 1
rr = r
w(ws, rr, 1, "Range across settings", bold=True)
w(ws, rr, 5, f"=MIN(E{r0}:E{r0+len(s16)-1})&\" to \"&MAX(E{r0}:E{r0+len(s16)-1})",  # noqa: E501
  font=CALC, bold=True)
w(ws, rr, 7, f"=MIN(G{r0}:G{r0+len(s16)-1})", DEC3, font=CALC, bold=True)
r = put_note(ws, r + 2,
             "Note. Each row rebuilds the instrument from the corpus "
             "vocabulary under that threshold triple and recomputes the "
             "city ranking. The thresholds change how much of the corpus the "
             "instrument sees — prevalence ranges from 32.1% to "
             "49.5% — but not which destinations it ranks high: the lowest "
             "rank correlation with the reported index is .93.", span=8)

w(ws, r, 1, "Panel C — terms falling just either side of the cuts", bold=True)
r += 1
r0 = put_header(ws, r, ["Term", "Similarity to hospitality",
                        "Similarity to praise", "Contrastive margin",
                        "Corpus frequency", "Excluded by"],
                [22, 16, 15, 14, 14, 20])
for i, row in enumerate(m16.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.term)
    w(ws, rr, 2, row.sim_hosp, DEC3)
    w(ws, rr, 3, row.sim_praise, DEC3)
    w(ws, rr, 4, f"=B{rr}-C{rr}", DEC3, font=CALC)
    w(ws, rr, 5, row.freq, NUM)
    w(ws, rr, 6, row.excluded_by)
r = r0 + len(m16) + 1
r = put_note(ws, r,
             "Note. The first block is terms that clear the contrastive "
             "margin but fall just under the similarity cut; the second "
             "is terms similar enough to the hospitality centroid but closer "
             "to generic praise. The second block is the "
             "contrastive filter doing its work: incrível, maravilhoso and "
             "excepcional are excluded here, and they become the "
             "generic-affect control used for discriminant validity.", span=6)

w(ws, r, 1, "Panel D — choosing the number of hospitality types", bold=True)
r += 1
r0 = put_header(ws, r, ["Number of types (k)", "Bootstrap adjusted Rand index",
                        "Silhouette width", "Selected"],
                [16, 20, 15, 12])
best = k16.bootstrap_ARI.idxmax()
for i, row in enumerate(k16.itertuples()):
    rr = r0 + i
    w(ws, rr, 1, row.k, NUM)
    w(ws, rr, 2, row.bootstrap_ARI, DEC3)
    w(ws, rr, 3, row.silhouette, DEC3)
    sel = "yes" if i == best else ""
    w(ws, rr, 4, sel, bold=bool(sel))
    if sel:
        for j in range(1, 5):
            ws.cell(row=rr, column=j).fill = NOTE_FILL
put_note(ws, r0 + len(k16) + 1,
         "Note. Stability is the mean adjusted Rand index between the full "
         "solution and 200 bootstrap resamples of the marker "
         "set, computed over their shared members. Silhouette widths are low "
         "and nearly flat across the whole range, which is "
         "itself the finding that the marker space is not composed of "
         "well-separated islands; selecting on separation would be "
         "choosing among differences that are not there.", span=4)

# ---- print setup: a reviewer who prints this should get something legible
for ws in wb.worksheets:
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_options.horizontalCentered = True
    ws.page_margins.left = ws.page_margins.right = 0.3
    ws.page_margins.top = ws.page_margins.bottom = 0.4
    header_row = None
    for rr_ in range(1, 12):
        if ws.cell(rr_, 1).fill and ws.cell(rr_, 1).fill.fgColor.rgb == "001F3864":  # noqa: E501
            header_row = rr_
            break
    if header_row:
        ws.print_title_rows = f"{header_row}:{header_row}"

wb.save(DEST)
print("saved", DEST)
