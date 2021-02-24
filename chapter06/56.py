import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(filename):
    with open(f'chapter06/data/processed/{filename}.features.txt') as f:
        df_true = pd.read_csv(f, sep='\t', header=None)
    df_true.columns = ['TITLE', 'CATEGORY'] + [i for i in range(len(df_true.columns) - 2)]

    with open(f'chapter06/outputs/{filename}.predicts.txt') as f:
        df_pred = pd.read_csv(f, sep='\t', header=None)
    df_pred.columns = ['predict']

    args = (df_true['CATEGORY'], df_pred['predict']) 
    scores = {
        'precision_micro': precision_score(*args, average='micro'),
        'precision_macro': precision_score(*args, average='macro'),
        'recall_micro': recall_score(*args, average='micro'),
        'recall_macro': recall_score(*args, average='macro'),
        'f1_micro': f1_score(*args, average='micro'),
        'f1_macro': f1_score(*args, average='macro')
    }
    return scores


print(evaluate('train'))
print(evaluate('valid'))
