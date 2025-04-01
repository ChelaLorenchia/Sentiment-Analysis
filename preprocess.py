import re
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 5000
MAX_LEN = 100

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Hanya huruf dan spasi
    return text

# Baca dataset
df = pd.read_csv("dataset\\Data1.csv", sep=";")
df['Cleaned_Comment'] = df['Comment'].astype(str).apply(clean_text)

# Tokenisasi
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Cleaned_Comment'])

# Simpan tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Konversi teks ke angka
sequences = tokenizer.texts_to_sequences(df['Cleaned_Comment'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

# Encode label
label_map = {"Negative": 0, "Netral": 1, "Positive": 2}
df['Label_Encoded'] = df['Label'].map(label_map)

df.to_csv("dataset/processed_data.csv", index=False)
