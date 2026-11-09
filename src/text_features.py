
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

df = pd.read_csv(OUTPUT / "reviews_with_sentiment.csv")

df = df.dropna(
    subset=["Review Text", "Recommended IND"]
).copy()

negative = df[df["Recommended IND"] == 0]
positive = df[df["Recommended IND"] == 1]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3
)

# Fit on all reviews to maintain one vocabulary
X = vectorizer.fit_transform(df["Review Text"])
terms = vectorizer.get_feature_names_out()

def get_top_terms(mask, top_n=25):
    scores = X[mask].mean(axis=0).A1

    result = pd.DataFrame({
        "term": terms,
        "mean_tfidf": scores
    })

    return result.sort_values(
        "mean_tfidf", ascending=False
    ).head(top_n)

negative_terms = get_top_terms(
    (df["Recommended IND"] == 0).to_numpy()
)

positive_terms = get_top_terms(
    (df["Recommended IND"] == 1).to_numpy()
)

print("\nTOP TERMS: NOT RECOMMENDED")
print(negative_terms)

print("\nTOP TERMS: RECOMMENDED")
print(positive_terms)

negative_terms.to_csv(
    OUTPUT / "negative_terms.csv", index=False
)

positive_terms.to_csv(
    OUTPUT / "positive_terms.csv", index=False
)
