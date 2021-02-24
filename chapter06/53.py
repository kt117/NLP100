import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd


def predict(model, filename):
    with open(f'chapter06/data/processed/{filename}.features.txt') as f:
        df = pd.read_csv(f, sep='\t', header=None)
    df.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df.columns) - 2)]

    X, y = df.drop(['TITLE', 'CATEGORY'], axis=1), df['CATEGORY']

    pd.DataFrame(model.predict(X)).to_csv(f'chapter06/outputs/{filename}.predicts.txt', sep='\t', index=False, header=False)
    pd.DataFrame(model.predict_proba(X)).to_csv(f'chapter06/outputs/{filename}.probas.txt', sep='\t', index=False, header=False)


with open('chapter06/models/classifier.pickle', 'rb') as f:
    model = pickle.load(f)

predict(model, 'train')
predict(model, 'valid')
