import time
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def load_dataloader(filename, batch_size):
    with open(f"chapter08/data/processed/X_{filename}.csv") as f:
        X = pd.read_csv(f, sep='\t')

    with open(f"chapter08/data/processed/y_{filename}.csv") as f:
        y = pd.read_csv(f, sep='\t')

    X = torch.tensor(np.array(X.drop("TITLE", axis=1).astype('f')))
    y = torch.tensor(np.array(y["CATEGORY"]))

    dataset = data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    return train_loss / size, correct / size


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return test_loss / size, correct / size


def train_model(batch_size=64):
    train_dataloader = load_dataloader("train", batch_size)
    valid_dataloader = load_dataloader("valid", batch_size)

    model = NeuralNetwork().to(device)

    learning_rate = 1e-1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    start = time.time()
    
    epochs = 200
    for epoch in tqdm(range(epochs)):
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)
    
    train_result = test_loop(train_dataloader, model, loss_fn)
    test_result = test_loop(valid_dataloader, model, loss_fn)
    print(train_result[1], test_result[1])


train_model()
