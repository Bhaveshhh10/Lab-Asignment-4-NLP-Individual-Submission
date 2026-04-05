import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'text']

# Show first 5 rows
print(df.head())

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess(text):
    # convert to lowercase
    text = text.lower()
    
    # remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # split into words
    words = text.split()
    
    # remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # stemming
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)

print(df[['text', 'clean_text']].head())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])

y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))