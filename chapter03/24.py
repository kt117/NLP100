import pickle
import re


def solve(text):
    pattern = re.compile(r'^\[\[File:.+\]\]$')
    res = list()
    for line in text.split('\n'):
        if re.match(pattern, line):
            res.append(line[7 : -2].split('|')[0])
    return res


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    res = solve(text)
    for file_name in res:
        print(file_name)