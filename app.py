from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import json
import re
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thesis-project.vercel.app"],  # Or use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================================================================
# Critical Fix for TF 2.10 + .keras Format
# =================================================================

# 1. Define custom metrics (MUST match training configuration)
class CustomMetrics:
    @staticmethod
    def prec(y_true, y_pred):
        return tf.keras.metrics.Precision(name='prec')(y_true, y_pred)
    
    @staticmethod
    def rec(y_true, y_pred):
        return tf.keras.metrics.Recall(name='rec')(y_true, y_pred)
    
    @staticmethod
    def auc(y_true, y_pred):
        return tf.keras.metrics.AUC(name='auc')(y_true, y_pred)

# 2. Load model with explicit custom objects
model = tf.keras.models.load_model(
    'best_model_NadamV1_work.keras',
    custom_objects={
        'prec': CustomMetrics.prec,
        'rec': CustomMetrics.rec,
        'auc': CustomMetrics.auc,
        'Nadam': tf.keras.optimizers.Nadam
    }
)

# =================================================================
# Load Profanity List
# =================================================================
# Assumes 'profanity_list.csv' is in the same folder
profanity_words = set(pd.read_csv('extended_profanity_list.csv', header=None)[0].str.lower())

def censor_text(text, profanity_words):
    words = text.split()
    censored_words = []
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation for checking
        if clean_word.lower() in profanity_words:
            censored_word = re.sub(r'\w', '*', word)  # Keep punctuation, censor letters
            censored_words.append(censored_word)
        else:
            censored_words.append(word)
    return ' '.join(censored_words)

# =================================================================
# Text Preprocessing Pipeline
# =================================================================
MAX_LEN = 500

# Load tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_config = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = str(text).replace("\n", " ")
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('[0-9]',"",text)
    text = re.sub(" +", " ", text)
    text = re.sub("([^\x00-\x7F])+"," ",text)
    return text

def preprocess(text):
    normalized = normalize_text(text)
    sequence = tokenizer.texts_to_sequences([normalized])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, 
        maxlen=MAX_LEN, 
        padding='post',
        truncating='post'
    )
    return padded

# =================================================================
# API Endpoints
# =================================================================
class TextRequest(BaseModel):
    text: str
# before threshold 0.62206481, 0.62625204, 0.62000381, 0.52563927, 0.30879551, 0.62308365
#optimized thresholds based on minimal difference of precision and recall 0.39773023, 0.42862859, 0.56014192, 0.44549420, 0.34632149, 0.64880210
THRESHOLDS = [0.39773023, 0.42862859, 0.56014192, 0.44549420, 0.34632149, 0.64880210]
LABELS = ['toxic', 'insult', 'profanity', 'threat', 'identity hate', 'very_toxic']

@app.post("/predict")
async def predict(request: TextRequest):
    # Preprocess input
    processed = preprocess(request.text)
    
    # Get predictions
    probabilities = model.predict(processed, verbose=0)[0]
    predictions = (probabilities > THRESHOLDS).astype(int)

    # Censor input text
    censored_text = censor_text(request.text, profanity_words)

    return {
        "original_text": request.text,
        "censored_text": censored_text,
        "probabilities": {label: float(prob) for label, prob in zip(LABELS, probabilities)},
        "predicted_labels": [LABELS[i] for i, val in enumerate(predictions) if val]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
