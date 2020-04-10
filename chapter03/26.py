import pickle
import re


class MarkupRemover:
    pattern = re.compile(r"(\'{2,5})(.*)(\1)")

    def remove(self, s):
        return re.sub(self.pattern, r"\2", s)


def solve(text):
    spattern = re.compile(r"^\{\{基礎情報")
    epattern = re.compile(r"^\}\}")
    fpattern = re.compile(r"^\|(.+?)=(.+)$")

    fields = dict()
    flag = False
    mr = MarkupRemover()
    for line in text.split('\n'):
        if re.match(spattern, line):
            flag = True
        if flag:
            res = re.match(fpattern, line)
            if res:
                fields[res.group(1).strip()] = mr.remove(res.group(2).strip())
        if re.match(epattern, line):
            flag = False
    return fields


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    fields = solve(text)
    for k, v in fields.items():
        print(k, v)