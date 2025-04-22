import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_WORDS = 5000
MAX_LEN = 100

# Load dataset
df = pd.read_csv("dataset/processed_data.csv")

# Konversi teks ke angka dengan tokenizer
sequences = tokenizer.texts_to_sequences(df['Cleaned_Comment'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

# Menampilkan beberapa Statement dan Sequence-nya
print("\n=== Contoh Statement dan Sequence ===\n")
for i in range(10):  # Ganti 10 jadi jumlah yang kamu mau
    statement = df['Cleaned_Comment'].iloc[i]
    tokens = tokenizer.texts_to_sequences([statement])[0]
    print(f"Statement: {statement}")
    print(f"Sequence: {tokens}")
    print()

# Pastikan tensor adalah `long`
train_texts = torch.tensor(padded_sequences, dtype=torch.long)
train_labels = torch.tensor(df['Label_Encoded'].values, dtype=torch.long)

# Dataset PyTorch
class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = CommentDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model LSTM
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

VOCAB_SIZE = MAX_WORDS
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

sample_input = torch.randint(0, MAX_WORDS, (1, MAX_LEN)).to(device)  # Simulasi input acak
print("\n=== Model Architecture & Layer Output Shapes ===")

x = sample_input
print(f"Input shape: {x.shape}")  # (1, 100)

x = model.embedding(x)
print(f"After Embedding: {x.shape}")  # (1, 100, 100)

output, (hidden, cell) = model.lstm(x)
print(f"After LSTM Output: {output.shape}")   # (1, 100, 128)
print(f"LSTM Hidden State: {hidden[-1].shape}")  # (1, 128)

x = model.fc(hidden[-1])
print(f"After Dense Layer: {x.shape}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Simpan model
torch.save(model.state_dict(), "model/model.pth")
print("Model berhasil disimpan di 'model/model.pth'")

# Plot
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluasi model setelah training
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Laporan evaluasi
print("\n======= Evaluasi Model ======")
print(classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive']))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("Accuracy:", accuracy_score(all_labels, all_preds))