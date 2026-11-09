
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reviews.csv"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nRECOMMENDATION DISTRIBUTION")
print(df["Recommended IND"].value_counts(normalize=True))

# Remove rows without usable review text or recommendation
df = df.dropna(
    subset=["Review Text", "Recommended IND"]
).copy()

df["Review Text"] = df["Review Text"].astype(str)
df["review_length"] = df["Review Text"].str.len()
df["word_count"] = df["Review Text"].str.split().str.len()

# Summarize recommendation rates by clothing category
category_stats = (
    df.groupby("Class Name", dropna=False)
    .agg(
        review_count=("Recommended IND", "size"),
        recommendation_rate=("Recommended IND", "mean"),
        average_rating=("Rating", "mean"),
        average_words=("word_count", "mean")
    )
    .sort_values("review_count", ascending=False)
)

print("\nCATEGORY STATISTICS")
print(category_stats)

category_stats.to_csv(OUTPUT / "category_statistics.csv")

# Plot recommendation rate by category
plot_data = category_stats[
    category_stats["review_count"] >= 30
].sort_values("recommendation_rate")

plt.figure(figsize=(11, 6))

sns.barplot(
    data=plot_data.reset_index(),
    x="recommendation_rate",
    y="Class Name"
)

plt.title("Recommendation Rate by Clothing Category")
plt.xlabel("Recommendation Rate")
plt.ylabel("Clothing Category")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT / "category_recommendation.png", dpi=150)
plt.close()

# Review length distribution
plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="word_count",
    hue="Recommended IND",
    bins=40,
    element="step"
)

plt.title("Review Length by Recommendation")
plt.xlabel("Number of Words")
plt.tight_layout()
plt.savefig(OUTPUT / "review_length.png", dpi=150)
plt.close()

print("\nEDA complete. Results saved to outputs/")
