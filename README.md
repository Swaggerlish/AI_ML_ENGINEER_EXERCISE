# AI/ML Engineer Exercise: Sentiment Analysis API

Dockerized FastAPI sentiment analysis service using TF-IDF Logistic Regression baseline (F1: 0.885 on 25k dataset).

## 🚀 Quick Start

### Local Development
```bash
conda create -n ai-exercise python=3.11
conda activate ai-exercise
pip install -r requirements.txt
python model_training.py
uvicorn app:app --reload --port 8000
# DOCKER
docker build -t ai-exercise .
docker run -p 8000:8000 ai-exercise
## API Reference
Interactive Docs: http://127.0.0.1:8000/docs
POST /predict
Analyze sentiment of text
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Love this product!"}'
Response:
{
  "prediction": "positive",
  "confidence": 0.954
}
# 🧠 ML Pipeline
## Model Architecture
Raw Text → TF-IDF Vectorizer (20k vocab)
         ↓
Logistic Regression (3 classes: pos/neu/neg)
         ↓
F1: 0.885 (20k train, 5k test)
## Training
# model_training.py
X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2)
vectorizer = TfidfVectorizer(max_features=20000)
model = LogisticRegression()

# Fit → F1: 0.885
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'baseline_model.joblib')
## DistilBERT (optional): Fine-tuning started, CPU-limited.

