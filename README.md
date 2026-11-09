# Fashion Review Analytics Using DistilBERT

A Natural Language Processing (NLP) project that uses **DistilBERT and Transformer-based deep learning** to analyze fashion product reviews and classify whether a customer recommends a product based on their written feedback.

## 1. Project Overview

Customer reviews contain valuable information about product quality, customer satisfaction, and purchasing behavior. However, manually analyzing thousands of reviews can be time-consuming and difficult to scale.

This project explores how Natural Language Processing and deep learning can automate the analysis of customer feedback.

Using a pretrained DistilBERT model, the project fine-tunes a transformer-based neural network to classify fashion reviews into two categories:

- **Recommended (1):** The customer recommends the product.
- **Not Recommended (0):** The customer does not recommend the product.

The goal is to develop a text classification model that can understand contextual information in customer reviews rather than relying only on individual keywords.

## 2. Project Objectives

- Develop an NLP-based classification system for fashion product reviews.
- Fine-tune a pretrained DistilBERT model for binary text classification.
- Understand how transformer architectures process natural language.
- Evaluate model performance using multiple classification metrics.
- Explore how customer feedback can support business intelligence and customer analytics.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | DistilBERT model and training |
| Hugging Face Datasets | Dataset preparation |
| Scikit-learn | Data splitting and model evaluation |
| DistilBERT | Transformer-based NLP classification |

## 4. Model Architecture

The project uses **DistilBERT (distilbert-base-uncased)**, a smaller and computationally efficient version of BERT.

DistilBERT uses a transformer architecture that learns contextual relationships between words, allowing it to understand how words contribute to the overall meaning of a sentence.

The pretrained model is fine-tuned for binary sequence classification.

### Workflow

1. Input customer review.
2. Tokenization using the DistilBERT tokenizer.
3. Contextual text representation through DistilBERT.
4. Classification layer produces two class scores.
5. Softmax converts scores into class probabilities.
6. Final prediction determines whether the product is recommended.

## 5. Dataset

The project uses a fashion review dataset containing customer feedback and recommendation labels.

Required dataset file:

`data/reviews.csv`

Required columns:

| Column | Description |
|---|---|
| Review Text | Written customer feedback |
| Recommended IND | Binary recommendation label (0 or 1) |

### Data Preprocessing

The preprocessing pipeline includes:

- Removing records with missing review text or recommendation labels.
- Converting review text into string format.
- Converting recommendation labels into integers.
- Removing empty reviews.
- Selecting the relevant text and target columns.

## 6. Model Training Strategy

The dataset is divided into three subsets:

| Dataset | Allocation | Purpose |
|---|---:|---|
| Training | 70% | Model learning and fine-tuning |
| Validation | 10% | Model evaluation during training |
| Testing | 20% | Final performance evaluation |

Stratified sampling is used to preserve the class distribution across the three subsets.

### Training Configuration

| Parameter | Value |
|---|---|
| Pretrained Model | DistilBERT |
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 2 |
| Maximum Sequence Length | 128 |
| Weight Decay | 0.01 |
| Random Seed | 42 |
| Model Selection Metric | Macro F1 |

The implementation also supports CPU, CUDA GPU, and Apple MPS device detection.

### Quick Test Mode

The project includes a quick testing configuration to make initial experimentation faster.

When enabled:
- Training uses up to 1,000 samples.
- Validation uses up to 200 samples.
- Testing uses up to 200 samples.

For full training, set:

`QUICK_TEST = False`

## 7. Model Evaluation

The model is evaluated using several classification metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| Precision | Measures how many predicted recommendations are correct |
| Recall | Measures how many actual recommendations are identified |
| F1-Score | Balances precision and recall |
| Macro F1 | Calculates the average F1-score across both classes |
| ROC-AUC | Measures the model's ability to distinguish between the two classes |

A classification report is also generated to provide class-level precision, recall, and F1-score.

Macro F1 is used to select the best model during training, giving equal weight to both recommendation classes.

## 8. Project Structure

```text
fashion-text-analytics/
│
├── data/
│   └── reviews.csv
│
├── models/
│   └── distilbert-fashion/
│       ├── checkpoints/
│       ├── quick-test/
│       └── ...
│
├── outputs/
│   ├── distilbert_quick_test_metrics.json
│   └── distilbert_test_metrics.json
│
├── src/
│   └── fashion-text-analytics.py
│
├── requirements.txt
│
└── README.md
```

The directory structure above represents the intended project organization. The training script creates the model and output directories automatically.

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or later recommended.
- VS Code or another Python development environment.
- Internet connection for downloading the pretrained model and tokenizer.
- Sufficient memory and storage for model training.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd fashion-text-analytics
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place your dataset inside the `data` directory:

```text
data/reviews.csv
```

Ensure that it contains the required `Review Text` and `Recommended IND` columns.

### Step 5: Run the Training Script

```bash
python src/fashion-text-analytics.py
```

The script will preprocess the dataset, tokenize the reviews, fine-tune DistilBERT, evaluate the model, and save the trained model and evaluation metrics.

## 10. Expected Outputs

After successful execution, the project generates:

**Trained Model**
- Fine-tuned DistilBERT model.
- Saved tokenizer.
- Training checkpoints.

**Evaluation Results**
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Macro F1.
- ROC-AUC.
- Detailed classification report.

**Metrics File**

Evaluation metrics are exported as JSON files in the `outputs/` directory.

## 11. Potential Future Improvements

This project provides a foundation for more advanced customer feedback analytics.

Potential extensions include:

1. **Sentiment Analysis:** Identify positive, negative, and neutral customer opinions.
2. **Aspect-Based Sentiment Analysis:** Identify specific product attributes discussed in reviews, such as fit, fabric, quality, comfort, and design.
3. **Customer Feedback Clustering:** Group reviews based on recurring themes and customer concerns.
4. **Model Explainability:** Investigate which words and phrases influence classification decisions.
5. **Model Deployment:** Build an API using FastAPI to serve predictions.
6. **Analytics Dashboard:** Develop an interactive dashboard to monitor customer feedback and recommendation patterns.
7. **Model Monitoring:** Track prediction performance and detect changes in customer language and review patterns over time.

## 12. Business Applications

The project demonstrates how NLP can transform unstructured customer feedback into structured information that supports business decisions.

Potential applications include:

- Understanding customer satisfaction.
- Identifying recurring product complaints.
- Supporting product quality improvements.
- Improving product recommendations.
- Monitoring customer feedback trends.
- Supporting merchandising and marketing strategies.

By automating review classification, businesses can process larger volumes of customer feedback and extract insights that may otherwise be difficult to identify manually.

## 13. Disclaimer

This is an educational and portfolio project developed to explore transformer-based NLP, deep learning, and customer feedback analytics.

The model's actual performance depends on the dataset, preprocessing, training configuration, and evaluation results. No specific performance level is assumed.

---

**Author:** Joshua Bloodymier Salvino

**Project Category:** Natural Language Processing | Deep Learning | Customer Analytics | Transformer Models

**Framework:** PyTorch and Hugging Face Transformers