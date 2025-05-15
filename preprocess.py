import re
import pickle
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 5000
MAX_LEN = 100

# ================== Load Kamus Slang ==================
slang_df = pd.read_csv("dataset/kamus_slang.csv", sep=";", encoding='utf-8')
slang_df.dropna(inplace=True)  # Hapus baris kosong jika ada
slang_dict = dict(zip(slang_df['slang'].astype(str), slang_df['formal'].astype(str)))

# ================== Siapkan Stopword Remover ==================
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()
factory_stem = StemmerFactory()  # <-- Definisikan factory_stem dulu
stemmer = factory_stem.create_stemmer()

# ================== Fungsi Preprocessing ==================
def normalize_slang(text, slang_dict):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Hanya huruf dan spasi
    text = normalize_slang(text, slang_dict)
    # Hapus stopword
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)  # Tambahkan stemming
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== TRAINING DATA ====================
print("Proses training data...")

df_train = pd.read_csv("dataset/Data1.csv", sep=";", encoding='utf-8')

# Buang baris dengan Comment kosong atau NaN sebelum preprocessing
df_train = df_train.dropna(subset=['Comment'])
df_train['Comment'] = df_train['Comment'].astype(str)  # Pastikan string

df_train['Cleaned_Comment'] = df_train['Comment'].apply(clean_text)

# Buang baris dengan Cleaned_Comment kosong (hasil clean_text '')
df_train = df_train[df_train['Cleaned_Comment'] != '']

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['Cleaned_Comment'])

# Simpan tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Konversi teks ke angka
sequences = tokenizer.texts_to_sequences(df_train['Cleaned_Comment'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Encode label
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df_train['Label_Encoded'] = df_train['Label'].map(label_map)

# Simpan data terproses
df_train.to_csv("dataset/processed_data.csv", index=False, encoding='utf-8')
print("Selesai menyimpan processed_data.csv")

# ===================== TESTING DATA =====================
print("Proses testing data...")

df_test = pd.read_csv("dataset/testing_data.csv", sep=";", encoding='utf-8')

# Buang baris kosong/NaN
df_test = df_test.dropna(subset=['Comment'])
df_test['Comment'] = df_test['Comment'].astype(str)

df_test['Cleaned_Comment'] = df_test['Comment'].apply(clean_text)
df_test = df_test[df_test['Cleaned_Comment'] != '']

# Load tokenizer dari training
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Gunakan tokenizer yang sudah dilatih
sequences = tokenizer.texts_to_sequences(df_test['Cleaned_Comment'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Encode label
df_test['Label_Encoded'] = df_test['Label'].map(label_map)

# Simpan data terproses
df_test.to_csv("dataset/processed_test_data.csv", index=False, encoding='utf-8')
print("Selesai menyimpan processed_test_data.csv")