import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def extract_features(filename, fit=False):
    with open(f'chapter06/data/processed/{filename}.txt') as f:
        df = pd.read_csv(f, sep='\t', header=None)
    df.columns = ['TITLE', 'CATEGORY']

    if fit:
        vectorizer = CountVectorizer(stop_words='english')
        bow = vectorizer.fit_transform(df['TITLE'])
        with open('chapter06/models/count_vectorizer.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        with open('chapter06/models/count_vectorizer.pickle', 'rb') as f:
            vectorizer = pickle.load(f)
        bow = vectorizer.transform(df['TITLE'])

    df = pd.concat([df, pd.DataFrame(bow.toarray())], axis=1)
    df.to_csv(f'chapter06/data/processed/{filename}.features.txt', sep='\t', index=False, header=None)


extract_features('train', fit=True)
extract_features('valid')
extract_features('test')
