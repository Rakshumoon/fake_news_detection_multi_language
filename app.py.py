import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk

# Download the stopwords corpus if not already downloaded
nltk.download('stopwords')

# Load the dataset
news_df = pd.read_csv('train.csv.zip')

# Fill NaN values with empty string
news_df = news_df.fillna(' ')

# Combine author and title into a single column
news_df['content'] = news_df['author'] + " " + news_df['title']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

# Apply stemming
news_df['content'] = news_df['content'].apply(stemming)

# Split data into features and labels
x = news_df['content'].values
y = news_df['label'].values

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
x_tfidf = tfidf_vectorizer.fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y, test_size=0.2, stratify=y, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Streamlit web app
st.title("Fake News Detection")

input_text = st.text_area("Enter news article", height=200)

if st.button("Predict"):
    if input_text:
        input_data = tfidf_vectorizer.transform([input_text])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("The news is fake")
        else:
            st.success("The news is real")
    else:
        st.warning("Please enter a news article to classify")
