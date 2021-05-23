import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


def load_data(filename):
    with open(f"chapter06/data/processed/{filename}.txt") as f:
        df = pd.read_csv(f, sep='\t', header=None)
    df.columns = ["TITLE", "CATEGORY"]
    return df


class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, max_len):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    text = self.X[index]
    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      pad_to_max_length=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    return {
      'ids': torch.LongTensor(ids),
      'mask': torch.LongTensor(mask),
      'labels': torch.Tensor(self.y[index])
    }


class BERTClass(torch.nn.Module):
  def __init__(self, drop_rate, output_size):
    super().__init__()
    self.bert = BertModel.from_pretrained("chapter09/models/bert-base-uncased")
    self.drop = torch.nn.Dropout(drop_rate)
    self.fc = torch.nn.Linear(self.bert.config.hidden_size, output_size)

  def forward(self, ids, mask):
    out = self.bert(ids, attention_mask=mask)
    out = self.fc(self.drop(out["pooler_output"]))
    return out


df_train = load_data("train")
df_valid = load_data("valid")
e = np.eye(4)
y_train = torch.Tensor([e[i] for i in df_train["CATEGORY"]])
y_valid = torch.Tensor([e[i] for i in df_valid["CATEGORY"]])

tokenizer = BertTokenizer.from_pretrained("chapter09/models/bert-base-uncased")
max_len = 20
dataset_train = CreateDataset(df_train["TITLE"], y_train, tokenizer=tokenizer, max_len=max_len)
dataset_valid = CreateDataset(df_valid["TITLE"], y_valid, tokenizer=tokenizer, max_len=max_len)


def calculate_loss_and_accuracy(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(ids, mask)

            loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for data in dataloader_train:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid, device)
        
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}') 


DROP_RATE = 0.4
OUTPUT_SIZE = 4
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

model = BERTClass(DROP_RATE, OUTPUT_SIZE)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, device=device)
