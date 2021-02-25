import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def load_data(filename):
    with open(f'chapter06/data/processed/{filename}.features.txt') as f:
        return pd.read_csv(f, sep='\t', header=None)


def train_lr(df, c):
    df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]
    X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']
    return LogisticRegression(random_state=42, C=c).fit(X, y)


def train_rf(df, d):
    df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]
    X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']
    return RandomForestClassifier(random_state=42, max_depth=d).fit(X, y)


def evaluate(model, df):
    df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]
    X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']
    return accuracy_score(y, model.predict(X))


data_names = ['train', 'valid', 'test']
datas = {data_name: load_data(data_name) for data_name in data_names}

best_score, best_algo, best_model = -1, None, None
for c in tqdm([0.25, 0.5, 1, 2, 4]):
    model = train_lr(datas['train'], c)
    valid_score = evaluate(model, datas['valid'])
    if valid_score > best_score:
        best_score, best_algo, best_model = valid_score, 'lr', model
        
for d in tqdm([2, 4, 6, 8]):
    model = train_rf(datas['train'], d)
    valid_score = evaluate(model, datas['valid'])
    if valid_score > best_score:
        best_score, best_algo, best_model = valid_score, 'rf', model

print(best_score, evaluate(best_model, datas['test']), best_algo)
