import pickle
import re


def solve(text):
    spattern = re.compile(r"^\{\{基礎情報")
    epattern = re.compile(r"^\}\}")
    fpattern = re.compile(r"^\|(.+?)=(.+)$")

    fields = dict()
    flag = False
    for line in text.split('\n'):
        if re.match(spattern, line):
            flag = True
        if flag:
            res = re.match(fpattern, line)
            if res:
                fields[res.group(1).strip()] = res.group(2).strip()
        if re.match(epattern, line):
            flag = False
    return fields


with open("chapter03/outputs/england.pickle", "rb") as f:
    text = pickle.load(f)
    fields = solve(text)
    for k, v in fields.items():
        print(k, v)