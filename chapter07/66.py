from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd


model = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)


def calculate(filename):
    with open(f"chapter07/data/wordsim353/{filename}.csv") as f:
        df = pd.read_csv(f)

    similarities = [model.similarity(df.at[i, "Word 1"], df.at[i, "Word 2"]) for i in tqdm(df.index)]
    return spearmanr(similarities, df["Human (mean)"])


for filename in ["set1", "set2", "combined"]:
    print(filename, calculate(filename))
