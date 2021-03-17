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


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return test_loss, correct / size


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("chapter08/models/model_weights.pth"))
model.eval()

loss_fn = nn.CrossEntropyLoss()

batch_size = 64
train_dataloader = load_dataloader("train", batch_size)
valid_dataloader = load_dataloader("valid", batch_size)

result_train = test_loop(train_dataloader, model, loss_fn)
result_valid = test_loop(valid_dataloader, model, loss_fn)

print(result_train[1])
print(result_valid[1])
