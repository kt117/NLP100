import pickle
import re


def solve(text):
    pattern = re.compile(r'^\[\[File:.+\]\]$')
    res = list()
    for line in text.split('\n'):
        if re.match(pattern, line):
            res.append(line[7 : -2].split('|')[0])
    return res


with open("chapter03/outputs/england.pickle", "rb") as f:
    text = pickle.load(f)
    file_names = solve(text)
    for file_name in file_names:
        print(file_name)