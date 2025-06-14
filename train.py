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

df = df.dropna(subset=['Label_Encoded'])
df['Label_Encoded'] = df['Label_Encoded'].astype(int)

# Tokenizer
sequences = tokenizer.texts_to_sequences(df['Cleaned_Comment'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

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

#Definisi Parameter
MAX_WORDS = 5000
MAX_LEN = 100

if __name__ == "__main__":
    
    print("\n=== Contoh Statement dan Sequence ===\n")
    for i in range(10):
        statement = df['Cleaned_Comment'].iloc[i]
        tokens = tokenizer.texts_to_sequences([statement])[0]
        print(f"Statement: {statement}")
        print(f"Sequence: {tokens}")
        print()

    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # === TRAINING ===
    df = pd.read_csv("dataset/processed_data.csv")
    df = df.dropna(subset=['Label_Encoded'])
    df['Label_Encoded'] = df['Label_Encoded'].astype(int)

    sequences = tokenizer.texts_to_sequences(df['Cleaned_Comment'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

    train_texts = torch.tensor(padded_sequences, dtype=torch.long)
    train_labels = torch.tensor(df['Label_Encoded'].values, dtype=torch.long)

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

    VOCAB_SIZE = MAX_WORDS
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSentiment(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
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

    torch.save(model.state_dict(), "model/model.pth")
    print("Model berhasil disimpan di 'model/model.pth'")

    # === EVALUASI ===
    test_df = pd.read_csv("dataset/processed_test_data.csv")
    test_df = test_df.dropna(subset=['Label_Encoded'])
    test_df['Label_Encoded'] = test_df['Label_Encoded'].astype(int)

    test_sequences = tokenizer.texts_to_sequences(test_df['Cleaned_Comment'])
    test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LEN)

    test_texts = torch.tensor(test_padded_sequences, dtype=torch.long)
    test_labels = torch.tensor(test_df['Label_Encoded'].values, dtype=torch.long)

    class CommentDatasetTest(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    test_dataset = CommentDatasetTest(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("\n======= Evaluasi Model di Data TEST =======")
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive']))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy_score(all_labels, all_preds))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Testing Data)')
    plt.tight_layout()
    plt.show()