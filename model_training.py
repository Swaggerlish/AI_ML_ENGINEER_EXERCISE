# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset, load_dataset
import joblib
from huggingface_hub import snapshot_download


dataset = load_dataset("imdb", split="train")
df = dataset.to_pandas()
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Baseline
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
baseline_model = LogisticRegression()
baseline_model.fit(X_train_tfidf, y_train)
baseline_pred = baseline_model.predict(X_test_tfidf)
print("Baseline F1:", f1_score(y_test, baseline_pred, average='weighted'))

# Save baseline
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(baseline_model, 'baseline_model.joblib')

# Advanced: DistilBERT (fine-tune snippet)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # pos/neg/neu

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

train_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train})).map(tokenize, batched=True)
# Similar for test; then Trainer...
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
trainer.save_model('distilbert_model')
