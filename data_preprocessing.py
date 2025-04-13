import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder

# Unduh NLTK data jika belum ada
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (pakai delimiter ; karena CSV-nya pakai titik koma)
df = pd.read_csv('dataset/Data1.csv', delimiter=';')

# Tampilkan kolom awal untuk verifikasi
print("Kolom tersedia:", df.columns)

# Buang baris kosong & strip spasi pada Label
df.dropna(subset=['Comment', 'Label'], inplace=True)
df['Label'] = df['Label'].astype(str).str.strip()

# # Cek label sebelum encode
# print("\nJumlah data per label (sebelum encode):")
# print(df['Label'].value_counts())

# Siapkan stopwords dan stemmer
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi pembersih teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)               # Hapus URL
    text = re.sub(r"[^\x00-\x7F]+", ' ', text)                        # Hapus emotikon / karakter non-ASCII
    text = re.sub(r'\d+', '', text)                                   # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]     # Hapus stopword
    tokens = [stemmer.stem(word) for word in tokens]                 # Stemming
    return ' '.join(tokens)

# Terapkan pembersihan ke kolom komentar
df['Cleaned_Comment'] = df['Comment'].apply(clean_text)

# Encode label
le = LabelEncoder()
df['Label_Encoded'] = le.fit_transform(df['Label'])

# Tampilkan mapping label
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nMapping Label (Label -> Angka):", label_mapping)

# Simpan hasil ke file baru
df.to_csv('dataset/processed_data.csv', index=False)
print("\n✅ Proses selesai! Data disimpan ke 'dataset/processed_data.csv'")
