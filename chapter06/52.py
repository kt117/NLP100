import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd


with open('chapter06/data/processed/train.features.txt') as f:
    df = pd.read_csv(f, sep='\t', header=None)
df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]

X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']

model = LogisticRegression(random_state=42)
model.fit(X, y)

with open('chapter06/models/classifier.pickle', 'wb') as f:
    pickle.dump(model, f)
