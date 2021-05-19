import pickle
from gensim.models import KeyedVectors
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import nltk
import numpy as np
import pandas as pd

np.random.seed(seed=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb =  nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden


class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])


with open("chapter09/models/word_to_id.pickle", 'rb') as f:
    word_to_id = pickle.load(f)


def tokenize(text, word_to_id=word_to_id):
    words = nltk.word_tokenize(text)
    return [word_to_id.get(word, 0) for word in words]


def load_dataset(filename):
    with open(f"chapter06/data/processed/{filename}.txt") as f:
        df_train = pd.read_csv(f, sep='\t', header=None)
    df_train.columns = ["TITLE", "CATEGORY"]

    X = [np.array(tokenize(title)) for title in df_train["TITLE"]]
    y = np.array(df_train["CATEGORY"])
    return CreateDataset(X, y)


def train_model(dataset, model, loss_fn, optimizer, batch_size, collate_fn):
    size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
def eval_model(dataset, model, loss_fn):
    size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=1)

    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"loss: {loss / size:>7f} accuracy: {correct / size:>7f}")


VOCAB_SIZE = len(set(word_to_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = VOCAB_SIZE - 1
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

vectors = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)
embeddings = np.zeros((VOCAB_SIZE, EMB_SIZE))
for word, id in word_to_id.items():
    if word in vectors:
        embeddings[id] = vectors[word]
    else:
        embeddings[id] = np.random.rand(EMB_SIZE)
embeddings = torch.tensor(embeddings).float()

model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, embeddings).to(device)


def collate_fn(batch):
    features, labels = list(zip(*batch))
    return pad_sequence(features, batch_first=True, padding_value=PADDING_IDX),  torch.tensor(labels)


train_dataset = load_dataset("train")
valid_dataset = load_dataset("valid")

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

batch_size = 64
epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_model(train_dataset, model, loss_fn, optimizer, batch_size, collate_fn)
    eval_model(train_dataset, model, loss_fn)
    eval_model(valid_dataset, model, loss_fn)
print("Done!")
