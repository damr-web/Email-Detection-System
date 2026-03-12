import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["email_db"]
collection = db["emails"]

# Fetch data
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

print("Data loaded from MongoDB:")
print(df.head())

# Features & labels
X = df["email_text"]
y = df["label"]

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("✅ Model and vectorizer saved successfully")