"""
Build a binary dataset: peptide treatment of inflammatory cells -> secreted IL-6.
Label 1 = IL-6 decreased; Label 0 = increased / no change / unclear.

Source: Europe PMC REST API (abstracts). Run locally:
    pip install requests pandas
    python build_peptide_il6_dataset.py
Output: peptide_il6_dataset.csv  (REVIEW the `needs_review` rows by hand)
"""

import re, time, requests, pandas as pd

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
QUERIES = [
    '(peptide) AND ("IL-6" OR interleukin-6) AND ELISA AND (macrophage OR RAW264.7 OR THP-1 OR microglia OR BV2 OR monocyte OR PBMC) AND (LPS OR inflammatory)',
    '(peptide) AND ("IL-6" OR interleukin-6) AND (ELISA OR "Luminex" OR "cytometric bead array") AND (keratinocyte OR fibroblast OR "dendritic cell" OR neutrophil OR synoviocyte)',
    '("anti-inflammatory peptide" OR "host defense peptide" OR "bioactive peptide" OR "antimicrobial peptide") AND "IL-6" AND ELISA',
]
PAGE, MAX_PAGES = 100, 10

DEC = r"(decreas|reduc|inhibit|suppress|attenuat|down-?regulat|lower|diminish|abolish|block|alleviat|ameliorat|dampen)"
INC = r"(increas|elevat|enhanc|induc|up-?regulat|promot|stimulat|boost|potentiat)"
NEG = r"(no (significant )?(effect|change)|did not (affect|alter|change)|unchanged|failed to)"
IL6 = r"(IL-?6|interleukin-?6)"
PEP = r"(peptide|hydrolysate|\b[A-Z][a-z]{1,3}-?[A-Z]{2,}\b)"
CELL = r"(macrophage|RAW\s?264|THP-?1|BV-?2|microglia|monocyte|PBMC|keratinocyte|HaCaT|dendritic|neutrophil|synoviocyte|fibroblast|cells?)"


def fetch():
    seen = {}
    for q in QUERIES:
        cursor = "*"
        for _ in range(MAX_PAGES):
            r = requests.get(
                API,
                params=dict(
                    query=q,
                    format="json",
                    resultType="core",
                    pageSize=PAGE,
                    cursorMark=cursor,
                ),
                timeout=60,
            )
            r.raise_for_status()
            j = r.json()
            for p in j.get("resultList", {}).get("result", []):
                if p.get("abstractText"):
                    seen[p.get("pmid") or p["id"]] = p
            nxt = j.get("nextCursorMark")
            if not nxt or nxt == cursor:
                break
            cursor = nxt
            time.sleep(0.4)
    return list(seen.values())


def label_sentence(s):
    """Return (label, needs_review) for one sentence mentioning IL-6."""
    dec = re.search(DEC, s, re.I)
    inc = re.search(INC, s, re.I)
    neg = re.search(NEG, s, re.I)
    if dec and not inc and not neg:
        return 1, False
    if (inc or neg) and not dec:
        return 0, False
    return (1 if dec else 0), True  # mixed signals -> flag


def build():
    rows = []
    for p in fetch():
        txt = re.sub(r"<[^>]+>", " ", p["abstractText"])
        for s in re.split(r"(?<=[.!?])\s+", txt):
            if (
                re.search(IL6, s, re.I)
                and re.search(PEP, s)
                and re.search(CELL, s, re.I)
            ):
                lab, rev = label_sentence(s)
                rows.append(
                    dict(
                        pmid=p.get("pmid"),
                        doi=p.get("doi"),
                        year=p.get("pubYear"),
                        title=p.get("title"),
                        evidence=s.strip(),
                        label=lab,
                        needs_review=rev,
                    )
                )
    df = pd.DataFrame(rows).drop_duplicates(subset=["pmid", "evidence"])
    df.to_csv("peptide_il6_dataset.csv", index=False)
    print(
        len(df),
        "rows;",
        df.label.value_counts().to_dict(),
        "| needs_review:",
        int(df.needs_review.sum()),
    )


if __name__ == "__main__":
    build()
