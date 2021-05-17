import pickle
from torch import nn
from torch.utils.data import Dataset
import torch
import nltk
import numpy as np
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
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


with open('chapter09/models/word_to_id.pickle', 'rb') as f:
    word_to_id = pickle.load(f)


def tokenize(text, word_to_id=word_to_id):
    words = nltk.word_tokenize(text)
    return [word_to_id.get(word, 0) for word in words]


with open("chapter06/data/processed/train.txt") as f:
    df_train = pd.read_csv(f, sep='\t', header=None)
df_train.columns = ["TITLE", "CATEGORY"]

X = [np.array(tokenize(title)) for title in df_train["TITLE"]]
y = np.array(df_train["CATEGORY"])
dataset = CreateDataset(X, y)

VOCAB_SIZE = len(set(word_to_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = VOCAB_SIZE - 1
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE).to(device)

logits = model(dataset[0][0].unsqueeze(0))
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print(pred_probab)
print(y_pred)
