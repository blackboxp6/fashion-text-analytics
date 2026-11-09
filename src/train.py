
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "reviews.csv")

df = df.dropna(subset=[
    "Review Text", "Recommended IND"
]).copy()

X = df["Review Text"].astype(str)
y = df["Recommended IND"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(
    y_test, probabilities
))

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "review_model.joblib")
