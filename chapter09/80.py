import pickle
from tqdm import tqdm
import nltk
import pandas as pd


def count_words(filename):
    with open(f"chapter06/data/processed/{filename}.txt") as f:
        df = pd.read_csv(f, sep='\t', header=None)
    df.columns = ["TITLE", "CATEGORY"]

    word_to_count = dict()
    for title in tqdm(df["TITLE"]):
        words = nltk.word_tokenize(title)
        for word in words:
            word_to_count[word] = word_to_count.get(word, 0) + 1
    return word_to_count


word_to_count = dict()
for filename in ["train", "valid", "test"]:
    word_to_count_file = count_words(filename)
    for word, count in word_to_count_file.items():
        word_to_count[word] = word_to_count.get(word, 0) + count

word_counts = sorted([(count, word) for word, count in word_to_count.items() if count >= 2], reverse=True)
word_to_id = {word_count[1]: i + 1 for i, word_count in enumerate(word_counts)}

with open('chapter09/models/word_to_id.pickle', 'wb') as f:
    pickle.dump(word_to_id, f)


def tokenize(text, word_to_id=word_to_id):
    words = nltk.word_tokenize(text)
    return [word_to_id.get(word, 0) for word in words]


print(tokenize("Fed official says weak data caused by weather, should not slow taper"))
