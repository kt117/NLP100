import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def load_data(filename):
    with open(f'chapter06/data/processed/{filename}.features.txt') as f:
        return pd.read_csv(f, sep='\t', header=None)


def train(df, c):
    df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]
    X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']
    return LogisticRegression(random_state=42, solver='sag', C=c).fit(X, y)


def evaluate(model, df):
    df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]
    X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']
    return accuracy_score(y, model.predict(X))


data_names = ['train', 'valid', 'test']
datas = {data_name: load_data(data_name) for data_name in data_names}

df_result = pd.DataFrame(columns=['c', 'key', 'score'])
for c in tqdm([0.25, 0.5, 1, 2, 4]):
    model = train(datas['train'], c)
    scores = {data_name: evaluate(model, df) for data_name, df in datas.items()}
    df_result = pd.concat([df_result, pd.DataFrame({'c': [c] * 3, 'key': data_names, 'score': [scores[data_name] for data_name in data_names]})])
df_result['c'] = np.log10(df_result['c'])

sns.scatterplot(data=df_result, x='c', y='score', hue='key')
plt.savefig('chapter06/outputs/58.png')
