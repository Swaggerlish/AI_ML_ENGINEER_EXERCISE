# train_models.py - Complete baseline + DistilBERT with evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification, 
    Trainer, TrainingArguments
)
import torch
from datasets import Dataset, load_dataset
import joblib

# 1. Load and split data
print("Loading IMDB dataset...")
dataset = load_dataset("imdb", split="train")
df = dataset.to_pandas()
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 2. TF-IDF Baseline
print("\nTraining TF-IDF baseline...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train_tfidf, y_train)
baseline_pred = baseline_model.predict(X_test_tfidf)
baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
print(f"Baseline F1: {baseline_f1:.4f}")
print(classification_report(y_test, baseline_pred))

# Save baseline
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(baseline_model, 'baseline_model.joblib')
print("Baseline saved.")

# 3. DistilBERT Fine-tuning
print("\nLoading DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2  # Binary: pos/neg (not 3)
)

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=256)

# Prepare datasets
print("Tokenizing datasets...")
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

test_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    eval_strategy='steps',  # Fixed deprecated warning
    eval_steps=500,
    logging_steps=100,
    report_to='none',  # No wandb/tensorboard
    dataloader_pin_memory=False,  # Fixed CPU warning
    warmup_steps=100,
    per_device_train_batch_size=8,  # Faster on CPU
    per_device_eval_batch_size=8,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("Starting DistilBERT training...")
trainer.train()
print("\nEvaluating...")
eval_results = trainer.evaluate()
print(f"DistilBERT F1: {eval_results['eval_f1']:.4f}, Acc: {eval_results['eval_accuracy']:.4f}")

# Save final model
trainer.save_model('distilbert_model')
tokenizer.save_pretrained('distilbert_model')
print("DistilBERT model saved to 'distilbert_model/'")
print("✅ Complete pipeline: Baseline F1 vs DistilBERT F1")
