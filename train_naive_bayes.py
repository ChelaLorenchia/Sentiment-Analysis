import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import pickle

# Load data yang sudah diproses
df = pd.read_csv('dataset/processed_data.csv')

# Hapus baris dengan komentar kosong
df.dropna(subset=['Cleaned_Comment'], inplace=True)

# Pisahkan fitur dan label
X = df['Cleaned_Comment']
y = df['Label_Encoded']

# Split data dengan stratifikasi label
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Inisialisasi dan latih model Naive Bayes
model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test_tfidf)
print("🔎 Akurasi:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Simpan model dan vectorizer
with open('model/naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("\n✅ Model dan vectorizer berhasil disimpan ke folder 'model/'")