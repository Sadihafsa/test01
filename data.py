import pandas as pd
df = pd.read_csv("emails_dataset.csv")
print(df.head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]
model = MultinomialNB()
model.fit(X, y)
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["email"]
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return jsonify({"classe": prediction[0]})
app.run(host="0.0.0.0", port=5000)
