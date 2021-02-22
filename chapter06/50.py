from sklearn.model_selection import train_test_split
import pandas as pd


with open('chapter06/data/NewsAggregatorDataset/newsCorpora.csv') as f:
    df = pd.read_csv(f, sep='\t', header=None)

df.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
df = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df = df[['TITLE', 'CATEGORY']]
df = df.sample(frac=1, random_state=17)

df_train, df_test =  train_test_split(df, test_size=0.2, random_state=17)
df_valid, df_test =  train_test_split(df_test, test_size=0.5, random_state=17)

df_train.to_csv('chapter06/output/train.txt', sep='\t', index=False, header=False)
df_valid.to_csv('chapter06/output/valid.txt', sep='\t', index=False, header=False)
df_test.to_csv('chapter06/output/test.txt', sep='\t', index=False, header=False)

print(df_train.groupby('CATEGORY').size())
print(df_valid.groupby('CATEGORY').size())
print(df_test.groupby('CATEGORY').size())
