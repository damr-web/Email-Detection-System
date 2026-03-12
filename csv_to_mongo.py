import pandas as pd
from pymongo import MongoClient

df = pd.read_csv("email_data.csv")

client = MongoClient("mongodb://localhost:27017/")
db = client["email_db"]
collection = db["emails"]

collection.delete_many({})
collection.insert_many(df.to_dict("records"))

print("✅ Email data stored in MongoDB")