from torch import nn
from torch.utils import data
import torch
import numpy as np
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(300, 4)

    def forward(self, x):
        logits = self.fc(x)
        return logits


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

with open("chapter08/data/processed/X_train.csv") as f:
    X = pd.read_csv(f, sep='\t')

with open("chapter08/data/processed/y_train.csv") as f:
    y = pd.read_csv(f, sep='\t')

X = torch.tensor(np.array(X.drop("TITLE", axis=1).astype('f')))
y = torch.tensor(np.array(y["CATEGORY"]))

model = NeuralNetwork().to(device)

logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

loss = nn.CrossEntropyLoss()
output = loss(pred_probab, y)
output.backward()
print(output)
print(model.fc.weight.grad)
