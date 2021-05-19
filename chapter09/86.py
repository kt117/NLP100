import pickle
from gensim.models import KeyedVectors
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import nltk
import numpy as np
import pandas as pd

np.random.seed(seed=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, embeddings=None):
        super().__init__()
        if embeddings is None:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        else:
            self.emb =  nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride=stride, padding=(padding, 0))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x.unsqueeze(1))
        conv = self.conv(emb)
        relu = self.relu(conv.squeeze(3))
        pool = F.max_pool1d(relu, relu.size()[2])
        out = self.fc(pool.squeeze(2))
        return out


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


VOCAB_SIZE = len(set(word_to_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = VOCAB_SIZE - 1
OUTPUT_SIZE = 4
OUT_CHANNELS = 100
KERNEL_HEIGHTS = 3
STRIDE = 1
PADDING = 1

vectors = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)
embeddings = np.zeros((VOCAB_SIZE, EMB_SIZE))
for word, id in word_to_id.items():
    if word in vectors:
        embeddings[id] = vectors[word]
    else:
        embeddings[id] = np.random.rand(EMB_SIZE)
embeddings = torch.tensor(embeddings).float()

model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, embeddings).to(device)

dataset = load_dataset("train")

logits = model(dataset[0][0].unsqueeze(0))
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print(pred_probab)
print(y_pred)
