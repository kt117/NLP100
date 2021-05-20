import pickle
from gensim.models import KeyedVectors
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import nltk
import numpy as np
import optuna
import pandas as pd

np.random.seed(seed=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, conv_params, drop_rate):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (kernel_height, emb_size), padding=(padding, 0)) for kernel_height, padding in conv_params])
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(len(conv_params) * out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x.unsqueeze(1))
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)) for conv in convs]
        pools_cat = torch.cat(pools, 1)
        out = self.fc(self.drop(pools_cat.squeeze(2)))
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


def load_dataset(filename, sort_by_length=False):
    with open(f"chapter06/data/processed/{filename}.txt") as f:
        df_train = pd.read_csv(f, sep='\t', header=None)
    df_train.columns = ["TITLE", "CATEGORY"]

    X = [np.array(tokenize(title)) for title in df_train["TITLE"]]
    y = np.array(df_train["CATEGORY"])

    if sort_by_length:
        index = np.argsort([len(v) for v in X])
        X = [X[i] for i in index]
        y = [y[i] for i in index]

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
    return loss / size, correct / size


def objective(trial):
    VOCAB_SIZE = len(set(word_to_id.values())) + 1
    PADDING_IDX = VOCAB_SIZE - 1
    OUTPUT_SIZE = 4
    CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
    EPOCHS = 30

    emb_size = int(trial.suggest_discrete_uniform('emb_size', 100, 300, 100))
    out_chanels = int(trial.suggest_discrete_uniform('out_chanels', 100, 500, 100))
    drop_rate = trial.suggest_discrete_uniform('drop_rate', 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    batch_size = 2 ** trial.suggest_int('log_batch_size', 4, 6)

    model = CNN(VOCAB_SIZE, emb_size, PADDING_IDX, OUTPUT_SIZE, out_chanels, CONV_PARAMS, drop_rate).to(device)

    def collate_fn(batch):
        features, labels = list(zip(*batch))
        return pad_sequence(features, batch_first=True, padding_value=PADDING_IDX),  torch.tensor(labels)

    train_dataset = load_dataset("train", sort_by_length=True)
    valid_dataset = load_dataset("valid")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(EPOCHS):
        train_model(train_dataset, model, loss_fn, optimizer, batch_size, collate_fn)

    loss_valid, _ = eval_model(valid_dataset, model, loss_fn)
    return loss_valid


study = optuna.create_study()
study.optimize(objective, n_trials=100)

trial = study.best_trial
print(f'loss: {trial.value:.7f}')
for key, value in trial.params.items():
    print(key, value)
