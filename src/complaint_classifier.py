
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

df = pd.read_csv(
    OUTPUT / "reviews_with_topics.csv"
)

# Rules require an issue phrase, not just a topic word.
COMPLAINT_PATTERNS = {
    "Sizing": [
        r"\btoo small\b",
        r"\btoo big\b",
        r"\bruns small\b",
        r"\bruns large\b",
        r"\bdoesn'?t fit\b",
        r"\bpoor fit\b",
        r"\bsizing (?:is )?inconsistent\b",
        r"\bsize (?:is )?wrong\b"
    ],
    "Fabric Quality": [
        r"\bcheap (?:fabric|material)\b",
        r"\b(?:fabric|material) (?:is |feels )?cheap\b",
        r"\bpoor quality\b",
        r"\btoo thin\b",
        r"\bsee.through\b",
        r"\bitchy\b",
        r"\bscratchy\b",
        r"\bflimsy\b"
    ],
    "Price": [
        r"\btoo expensive\b",
        r"\boverpriced\b",
        r"\bnot worth (?:the )?(?:price|money)\b",
        r"\bprice (?:is )?too high\b"
    ],
    "Delivery": [
        r"\blate delivery\b",
        r"\barrived late\b",
        r"\bshipping delay\b",
        r"\bpackage never arrived\b",
        r"\bslow shipping\b"
    ],
    "Design": [
        r"\bpoor design\b",
        r"\bunflattering\b",
        r"\blooks ugly\b",
        r"\bbadly designed\b",
        r"\bawkward cut\b"
    ]
}

COMPILED = {
    category: [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in patterns
    ]
    for category, patterns in COMPLAINT_PATTERNS.items()
}

def classify_complaints(text):
    if pd.isna(text):
        return []

    text = str(text)

    return [
        category
        for category, patterns in COMPILED.items()
        if any(p.search(text) for p in patterns)
    ]

df["complaints"] = df["Review Text"].apply(
    classify_complaints
)

for category in COMPILED:
    df[category] = df["complaints"].apply(
        lambda labels: int(category in labels)
    )

df["complaint_count"] = df[
    list(COMPILED.keys())
].sum(axis=1)

print("\nCOMPLAINT COUNTS")
print(df[list(COMPILED.keys())].sum())

print("\nREVIEWS WITH DETECTED COMPLAINTS")
print((df["complaint_count"] > 0).sum())

df["complaints"] = df["complaints"].apply(
    lambda labels: ", ".join(labels)
)

df.to_csv(
    OUTPUT / "reviews_with_complaints.csv",
    index=False
)

print("\nComplaint classification completed.")
