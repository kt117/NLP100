import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm
import nltk
import numpy as np
import pandas as pd


model = KeyedVectors.load_word2vec_format("chapter07/data/GoogleNews-vectors-negative300.bin", binary=True)


def calculate_features(filename):
    with open(f"chapter06/data/processed/{filename}.txt") as f:
        df = pd.read_csv(f, sep='\t', header=None)
    df.columns = ["TITLE", "CATEGORY"]
    X, y = df[["TITLE"]], df["CATEGORY"]

    matrix = list()
    for title in tqdm(df["TITLE"]):
        words = nltk.word_tokenize(title)
        vectors = np.array([model[word] for word in words if word in model.vocab])
        matrix.append(np.mean(vectors, axis=0))
    X = pd.concat([X, pd.DataFrame(matrix)], axis=1)
    X.to_csv(f"chapter08/data/processed/X_{filename}.csv", sep='\t', index=False)
    y.to_csv(f"chapter08/data/processed/y_{filename}.csv", sep='\t', index=False)


for filename in ["train", "valid", "test"]:
    calculate_features(filename)
