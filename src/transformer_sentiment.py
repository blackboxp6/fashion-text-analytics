
from transformers import pipeline

MODEL_NAME = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_model = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1
)

def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "label": "UNKNOWN",
            "score": 0.0
        }

    result = sentiment_model(
        text,
        truncation=True,
        max_length=512
    )[0]

    return {
        "label": result["label"],
        "score": float(result["score"])
    }

if __name__ == "__main__":
    examples = [
        "I absolutely love this dress!",
        "The fabric is cheap and uncomfortable.",
        "The design is beautiful but the fit is terrible."
    ]

    for review in examples:
        print("\nReview:", review)
        print(analyze_sentiment(review))
