# 📰 Fake News Detection Web App

A machine learning-based web app to detect whether a news article is **fake** or **real**, built using Streamlit and Logistic Regression.

## 🚀 Features

- NLP preprocessing with stemming and stopword removal
- TF-IDF vectorization of news content
- Real-time prediction of news authenticity
- Logistic Regression model
- Interactive UI using Streamlit

## 🗂 Dataset

Uses the `train.csv` dataset from the [Fake News Dataset on Kaggle](https://www.kaggle.com/c/fake-news/data).

**Make sure to include `train.csv.zip` in the project folder.**

## 🧰 Tech Stack

- Python
- Scikit-learn
- Pandas, Numpy
- Streamlit
- NLTK

## 📦 How to Run

1. Install requirements:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
