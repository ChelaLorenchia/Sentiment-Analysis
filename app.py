from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan TF-IDF vectorizer
with open('model/naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Prediksi sentimen berdasarkan input
def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    sentiment = ["Negative", "Neutral", "Positive"]
    return sentiment[prediction[0]]

# Route untuk halaman utama
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        comment = request.form["comment"]
        sentiment = predict_sentiment(comment)
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
