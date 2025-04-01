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
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
