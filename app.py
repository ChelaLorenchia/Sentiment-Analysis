from flask import Flask, request, jsonify, render_template

import torch
import pickle
from torch.nn.functional import softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train import LSTMSentiment, MAX_WORDS, MAX_LEN

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(MAX_WORDS, 100, 128, 3).to(device)
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.eval()

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    tensor = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = softmax(output, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[prediction], probabilities.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["comment"]
        sentiment, prob = predict_sentiment(text)
        return render_template("index.html", text=text, sentiment=sentiment, prob=prob)
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "")
    sentiment, prob = predict_sentiment(text)
    return jsonify({"text": text, "sentiment": sentiment, "probabilities": prob})

if __name__ == "__main__":
    app.run(debug=True)
