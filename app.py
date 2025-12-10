# app.py
import os
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Flask App Setup
app = Flask(__name__, static_folder="static", template_folder="static")

# Load Classical Model (TF-IDF)
TFIDF_MODEL_PATH = os.path.join("models", "tfidf_model.pkl")

tfidf_data = joblib.load(TFIDF_MODEL_PATH)
vectorizer = tfidf_data["vectorizer"]
clf = tfidf_data["clf"]

print("Loaded classical TF-IDF model.")

# Load BERT Model (local folder)
BERT_DIR = os.path.join("models", "bert_model")

tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
bert_model.eval()

print("Loaded local BERT model.")

# Serve Frontend
@app.route("/")
def index_page():
    return send_from_directory("static", "index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # CLASSICAL MODEL
    text_vec = vectorizer.transform([text])
    classical_pred = clf.predict(text_vec)[0]
    classical_prob = None

    try:
        classical_prob = clf.predict_proba(text_vec)[0].tolist()
    except:
        pass

    # BERT MODEL
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        bert_pred_idx = int(np.argmax(probs))

    # Map BERT output labels (if using SST-2)
    bert_label_map = {0: "negative", 1: "positive"}
    bert_pred = bert_label_map.get(bert_pred_idx, "unknown")

    response = {
        "classical": {
            "prediction": classical_pred,
            "probabilities": classical_prob
        },
        "bert": {
            "prediction": bert_pred,
            "probabilities": probs.tolist()
        }
    }

    return jsonify(response)

# Start Flask
if __name__ == "__main__":
    app.run(debug=True)