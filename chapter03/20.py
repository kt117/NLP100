import json
import pickle

def solve(f):
    for line in f:
        article = json.loads(line[: -1])
        if article["title"] == "イギリス":
            return article


with open("data/jawiki-country.json") as f, open("output/article.pickle", mode='wb') as g:
    article = solve(f)
    print(article)
    pickle.dump(article, g)