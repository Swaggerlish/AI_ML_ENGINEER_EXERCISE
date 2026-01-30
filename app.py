# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import torch
from transformers import pipeline

app = FastAPI(title="Sentiment API")

# Load models
vectorizer = joblib.load('vectorizer.joblib')
baseline_model = joblib.load('baseline_model.joblib')
#sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert_model', return_all_scores=True)

sentiment_pipeline = None
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(input: TextInput):
    try:
        vec = vectorizer.transform([input.text])
        pred = baseline_model.predict(vec)[0]
        labels = ['negative', 'neutral', 'positive']
        return {"sentiment": labels[pred]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(input: BatchInput):
    try:
        vecs = vectorizer.transform(input.texts)
        preds = baseline_model.predict(vecs)
        labels = ['negative', 'neutral', 'positive']
        return [{"sentiment": labels[p]} for p in preds]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
