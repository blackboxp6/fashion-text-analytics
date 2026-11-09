
from pathlib import Path
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reviews.csv"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

df = df.dropna(subset=["Review Text"]).copy()
df["Review Text"] = df["Review Text"].astype(str)

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return pd.Series({
        "sentiment_score": compound,
        "sentiment_label": label
    })

df[["sentiment_score", "sentiment_label"]] = (
    df["Review Text"].apply(analyze_sentiment)
)

print("\nSENTIMENT DISTRIBUTION")
print(df["sentiment_label"].value_counts())

print("\nSENTIMENT BY CLOTHING CATEGORY")
print(
    pd.crosstab(
        df["Class Name"],
        df["sentiment_label"],
        normalize="index"
    ).round(3)
)

df.to_csv(OUTPUT / "reviews_with_sentiment.csv", index=False)

print("\nSaved sentiment results.")
