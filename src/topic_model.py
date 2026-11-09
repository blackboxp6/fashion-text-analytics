
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(
    OUTPUT / "reviews_with_sentiment.csv"
)

df = df.dropna(subset=["Review Text"]).copy()
df["Review Text"] = df["Review Text"].astype(str)

# Step 1: Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.90,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["Review Text"])

# Step 2: Train topic model
N_TOPICS = 6

nmf = NMF(
    n_components=N_TOPICS,
    init="nndsvda",
    random_state=42,
    max_iter=500
)

W = nmf.fit_transform(X)
H = nmf.components_

terms = vectorizer.get_feature_names_out()

# Step 3: Extract top keywords per topic
topic_rows = []

for topic_id, weights in enumerate(H):
    top_indices = np.argsort(weights)[::-1][:15]

    keywords = [terms[i] for i in top_indices]

    print(f"\nTOPIC {topic_id}")
    print(", ".join(keywords))

    for rank, i in enumerate(top_indices, start=1):
        topic_rows.append({
            "topic_id": topic_id,
            "rank": rank,
            "keyword": terms[i],
            "weight": float(weights[i])
        })

pd.DataFrame(topic_rows).to_csv(
    OUTPUT / "topic_keywords.csv",
    index=False
)

# Step 4: Assign dominant topic to each review
df["dominant_topic"] = W.argmax(axis=1)
df["topic_weight"] = W.max(axis=1)

df.to_csv(
    OUTPUT / "reviews_with_topics.csv",
    index=False
)

# Step 5: Save model and vectorizer
joblib.dump(
    {
        "vectorizer": vectorizer,
        "nmf": nmf
    },
    MODELS / "nmf_topic_model.joblib"
)

print("\nTopic modeling completed.")
