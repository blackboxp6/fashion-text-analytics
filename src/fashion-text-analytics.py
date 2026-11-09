
from pathlib import Path
import inspect
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "distilbert-fashion"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert/distilbert-base-uncased"

# Set False when ready for full training
QUICK_TEST = True

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 2
RANDOM_STATE = 42

# ==================================================
# 2. DEVICE DETECTION
# ==================================================

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"\nUsing device: {device}")

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No CUDA GPU detected. Training may be slower.")

# ==================================================
# 3. LOAD DATASET
# ==================================================

df = pd.read_csv(ROOT / "data" / "reviews.csv")

df = df.dropna(
    subset=["Review Text", "Recommended IND"]
).copy()

df["text"] = df["Review Text"].astype(str)
df["labels"] = df["Recommended IND"].astype(int)

df = df[df["text"].str.strip() != ""]

df = df[["text", "labels"]]

print("\nDataset size:", len(df))
print("\nClass distribution:")
print(df["labels"].value_counts())

# ==================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ==================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["labels"],
    random_state=RANDOM_STATE
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.125,
    stratify=train_df["labels"],
    random_state=RANDOM_STATE
)

# 70% training / 10% validation / 20% testing

if QUICK_TEST:

    print("\nQUICK TEST MODE ENABLED")

    train_df, _ = train_test_split(
        train_df,
        train_size=min(1000, len(train_df)),
        stratify=train_df["labels"],
        random_state=RANDOM_STATE
    ) if len(train_df) > 1000 else (train_df, None)

    val_df, _ = train_test_split(
        val_df,
        train_size=min(200, len(val_df)),
        stratify=val_df["labels"],
        random_state=RANDOM_STATE
    ) if len(val_df) > 200 else (val_df, None)

    test_df, _ = train_test_split(
        test_df,
        train_size=min(200, len(test_df)),
        stratify=test_df["labels"],
        random_state=RANDOM_STATE
    ) if len(test_df) > 200 else (test_df, None)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Testing samples:", len(test_df))

# ==================================================
# 5. TOKENIZATION
# ==================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

def create_dataset(dataframe):
    dataset = Dataset.from_pandas(
        dataframe.reset_index(drop=True)
    )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    return dataset.remove_columns(["text"])

train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)

# Automatically pad variable-length sequences
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)

# ==================================================
# 6. LOAD PRETRAINED DISTILBERT
# ==================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NOT_RECOMMENDED",
        1: "RECOMMENDED"
    },
    label2id={
        "NOT_RECOMMENDED": 0,
        "RECOMMENDED": 1
    }
)

# ==================================================
# 7. EVALUATION METRICS
# ==================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1
    ).numpy()[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            labels, predictions
        ),
        "precision": precision_score(
            labels, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels, predictions,
            zero_division=0
        ),
        "f1": f1_score(
            labels, predictions,
            zero_division=0
        ),
        "macro_f1": f1_score(
            labels, predictions,
            average="macro",
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            labels, probabilities
        )
    }

# ==================================================
# 8. TRAINING CONFIGURATION
# ==================================================

kwargs = dict(
    output_dir=str(MODEL_DIR / "checkpoints"),

    learning_rate=2e-5,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    save_strategy="epoch",

    logging_steps=20,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    report_to="none",

    fp16=(device == "cuda"),

    dataloader_pin_memory=(device == "cuda"),

    save_total_limit=2,

    seed=RANDOM_STATE
)

parameters = inspect.signature(
    TrainingArguments.__init__
).parameters

if "eval_strategy" in parameters:
    kwargs["eval_strategy"] = "epoch"
else:
    kwargs["evaluation_strategy"] = "epoch"

training_args = TrainingArguments(**kwargs)

# ==================================================
# 9. INITIALIZE TRAINER
# ==================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ==================================================
# 10. TRAIN MODEL
# ==================================================

print("\nStarting DistilBERT fine-tuning...")

trainer.train()

print("\nTraining completed!")

# ==================================================
# 11. FINAL EVALUATION
# ==================================================

print("\nEvaluating on test dataset...")

results = trainer.evaluate(test_ds)

print("\nFINAL TEST METRICS")

for key, value in results.items():
    if isinstance(value, (float, int)):
        print(f"{key}: {value:.4f}")

# Detailed classification report
pred_output = trainer.predict(test_ds)

predictions = np.argmax(
    pred_output.predictions,
    axis=1
)

labels = pred_output.label_ids

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Not Recommended",
            "Recommended"
        ],
        zero_division=0
    )
)

# ==================================================
# 12. SAVE TRAINED MODEL
# ==================================================

# Keep quick-test and full models separate
save_dir = (
    MODEL_DIR / "quick-test"
    if QUICK_TEST
    else MODEL_DIR
)

save_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(save_dir))
tokenizer.save_pretrained(str(save_dir))

# Save evaluation metrics
metrics_path = OUTPUT_DIR / (
    "distilbert_quick_test_metrics.json"
    if QUICK_TEST
    else "distilbert_test_metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(results, file, indent=4)

print("\nModel saved to:", save_dir)
print("Metrics saved to:", metrics_path)

print("\nFINISHED SUCCESSFULLY")
