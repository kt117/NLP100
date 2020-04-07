import pickle
import re


def solve(text):
    pattern = re.compile(r'^\[\[Category:.+\]\]$')
    res = list()
    for line in text.split('\n'):
        if re.match(pattern, line):
            res.append(line[11 : -2].split('|')[0])
    return res


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    category_names = solve(text)
    for category_name in category_names:
        print(category_name)