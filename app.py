import streamlit as st
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Spam Detector", layout="centered")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.main {
    background: transparent;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}
.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}
.stButton button {
    width: 100%;
    border-radius: 10px;
    background-color: #3b82f6;
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="title">📩 Spam Message Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered message classification</div>', unsafe_allow_html=True)

# ---------- LOAD DATA ----------
nltk.download('stopwords', quiet=True)

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------- MODEL ----------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# ---------- UI ----------
user_input = st.text_area("✍️ Enter your message here")

if st.button("🔍 Check Message"):
    cleaned = preprocess(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.error("🚨 This is SPAM")
    else:
        st.success("✅ This is NOT SPAM")