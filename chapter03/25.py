import pickle
import re


def solve(text):
    spattern = re.compile(r"^\{\{基礎情報")
    epattern = re.compile(r"^\}\}")

    res = dict()
    flag = False
    for line in text.split('\n'):
        if re.match(spattern, line):
            flag = True
        if flag and line[0] == '|':
            field = line.split('=')
            res[field[0][1 :].strip()] = field[1].strip()
        if re.match(epattern, line):
            flag = False
    return res


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    fields = solve(text)
    for k, v in fields.items():
        print(k, "->", v)