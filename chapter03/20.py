import json
import pickle

def solve(f):
    for line in f:
        article = json.loads(line[: -1])
        if article["title"] == "イギリス":
            return article["text"]


with open("data/jawiki-country.json") as f, open("output/england.pickle", mode='wb') as g:
    text = solve(f)
    pickle.dump(text, g)