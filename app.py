from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/spam_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email_text"]
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]

    result = "🚫 Spam Email" if prediction == "spam" else "✅ Genuine Email"
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)