from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model dan vectorizer
with open('model/naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Mapping label (ubah sesuai hasil LabelEncoder)
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    prob = None
    text = ''
    
    if request.method == 'POST':
        text = request.form.get('comment', '').strip()
        print("Komentar:", text)  # Debug

        if text:
            try:
                transformed = vectorizer.transform([text])
                prediction = model.predict(transformed)[0]
                prediction_prob = model.predict_proba(transformed).max()

                sentiment = label_map.get(prediction, "Tidak Diketahui")
                prob = f"{prediction_prob:.2f}"
            except Exception as e:
                print("Error saat prediksi:", e)
                sentiment = "Terjadi kesalahan"
                prob = "-"
        else:
            sentiment = "Komentar kosong"
            prob = "-"
    
    return render_template('index.html', text=text, sentiment=sentiment, prob=prob)

if __name__ == '__main__':
    app.run(debug=True)
