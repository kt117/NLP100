from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(300, 4)

    def forward(self, x):
        logits = self.fc(x)
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
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()


batch_size = 64
train_dataloader = load_dataloader("train", batch_size)
valid_dataloader = load_dataloader("valid", batch_size)

model = NeuralNetwork().to(device)

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(valid_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "chapter08/models/model_weights.pth")
