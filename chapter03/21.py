import pickle
import re


def solve(text):
    pattern = re.compile(r'^\[\[Category:.+\]\]$')
    res = list()
    for line in text.split('\n'):
        if re.match(pattern, line):
            res.append(line + '\n')
    return res


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    categories = solve(text)
    for category in categories:
        print(category)