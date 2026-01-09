# SmartWatch Sentiment Analyser

A Generative AI and Machine Learning based project that analyzes smartwatch
product reviews and predicts sentiment (Positive / Neutral / Negative).
The system uses two different models — a classical Logistic Regression
model and a transformer-based BERT model — for comparison and prediction.

# Problem Statement

Smartwatch platforms receive a huge number of customer reviews every day.
Manually analyzing these reviews to understand user sentiment is difficult
and inefficient. This project automates sentiment analysis using Natural
Language Processing (NLP) techniques and machine learning models.

# Objectives

To classify smartwatch reviews into sentiment categories

To compare traditional ML and advanced transformer models

To demonstrate the use of Generative AI in text understanding

To build a simple web application for real-time prediction

# Models Used
# 1 Logistic Regression (Classical ML)

Used with TF-IDF / Count Vectorizer

Fast, lightweight, and interpretable

Serves as a baseline model

# 2️ BERT (Transformer-based GenAI Model)

Bidirectional Encoder Representations from Transformers

Captures deep contextual meaning of text

Loaded using HuggingFace Transformers

Example Model:

nlptown/bert-base-multilingual-uncased-sentiment

# Model Note

Pretrained BERT model files are not stored in this repository due to GitHub
file size limitations. The model is automatically downloaded from
HuggingFace at runtime.
The Logistic Regression model is trained using the provided dataset.

# Tech Stack

Python

Flask

HuggingFace Transformers

Scikit-learn

Pandas, NumPy

HTML, CSS, JavaScript

📂 Project Structure
SmartWatch-Sentiment-Analyser
│
├── app.py
├── Gen_AI_Project.ipynb
├── data.xlsx
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── models/          # (ignored: model files not pushed)
├── .gitignore
└── README.md

#  How to Run the Project
Step 1: Clone the repository
git clone https://github.com/sarvani10/SmartWatch-Sentiment-Analyser.git
cd SmartWatch-Sentiment-Analyser

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Run the application
python app.py

Step 4: Open in browser
http://127.0.0.1:5000/

# Dataset

Dataset contains smartwatch review text and sentiment labels

Stored in data.xlsx

Used for both:

Training Logistic Regression

Testing BERT predictions

# Features

Dual-model sentiment analysis

Comparison of ML vs GenAI performance

Real-time user input prediction

Web-based interface using Flask

Clean and modular project structure

# Workflow Overview

User enters a smartwatch review

Text is preprocessed using NLP techniques

Review is passed to:

Logistic Regression model

BERT model

Predicted sentiment is displayed

# Contributors
- Sarvani Gogireddy  
- Mahathi Popuri  
- Neelima Lakshisetty
- Lahari
- Yasa sri


# Future Enhancements

Fine-tune BERT on smartwatch-specific data

Add model accuracy comparison dashboard

Deploy on cloud (AWS / GCP / Heroku)

Support multilingual reviews

# Conclusion

This project demonstrates the effectiveness of combining traditional
machine learning and Generative AI transformer models for sentiment
analysis. It highlights how advanced NLP models like BERT outperform
classical methods while also showing the importance of baseline approaches
such as Logistic Regression.

