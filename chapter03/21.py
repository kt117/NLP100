import pickle
import re


def solve(text):
    pattern = re.compile(r'^\[\[Category:.+\]\]$')
    res = ""
    for line in text.split('\n'):
        if re.match(pattern, line):
            res += line + '\n'
    return res


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    print(solve(text))