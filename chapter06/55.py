import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate(filename):
    with open(f'chapter06/data/processed/{filename}.features.txt') as f:
        df_true = pd.read_csv(f, sep='\t', header=None)
    df_true.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df_true.columns) - 2)]

    with open(f'chapter06/outputs/{filename}.predicts.txt') as f:
        df_pred = pd.read_csv(f, sep='\t', header=None)
    df_pred.columns = ['predict']

    return confusion_matrix(df_true['CATEGORY'], df_pred['predict'])


print(evaluate('train'))
print(evaluate('valid'))
