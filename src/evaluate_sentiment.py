
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "reviews.csv")

df = df.dropna(subset=["Review Text", "Rating"]).copy()

# Proxy labels based on review ratings
df = df[
    (df["Rating"] <= 2) |
    (df["Rating"] >= 4)
].copy()

df["target"] = (df["Rating"] >= 4).astype(int)

# Stratified sample to keep evaluation practical
from sklearn.model_selection import train_test_split

if len(df) > 500:
    _, df = train_test_split(
        df,
        test_size=500,
        stratify=df["target"],
        random_state=42
    )

df = df.reset_index(drop=True)

# VADER baseline
vader = SentimentIntensityAnalyzer()

df["vader_pred"] = df["Review Text"].apply(
    lambda text: int(
        vader.polarity_scores(str(text))["compound"] >= 0.05
    )
)

# Transformer predictions
model_name = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

classifier = pipeline(
    "text-classification",
    model=model_name,
    device=-1
)

texts = df["Review Text"].astype(str).tolist()

results = classifier(
    texts,
    truncation=True,
    max_length=512,
    batch_size=8
)

df["transformer_pred"] = [
    int(result["label"] == "POSITIVE")
    for result in results
]

for name, column in [
    ("VADER", "vader_pred"),
    ("DistilBERT", "transformer_pred")
]:
    print(f"\n{name}")

    print("Accuracy:", accuracy_score(
        df["target"], df[column]
    ))

    print("Macro F1:", f1_score(
        df["target"],
        df[column],
        average="macro"
    ))

    print(classification_report(
        df["target"],
        df[column],
        target_names=["Negative", "Positive"]
    ))

df.to_csv(
    OUTPUT / "transformer_evaluation.csv",
    index=False
)
