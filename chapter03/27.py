import pickle
import re


class MarkupRemover:
    patterns = [
        r"(\'{2,5})(.*)(\1)",
        r"\[\[([^\[\|]*?)\|?([^\[\|]*)\]\]"
    ]
    patterns = [re.compile(p) for p in patterns]
    poss = [r"\2", r"\2"]

    def remove(self, s):
        for p, pos in zip(self.patterns, self.poss):
            s = re.sub(p, pos, s)
        return s


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


with open("chapter03/outputs/england.pickle", "rb") as f:
    text = pickle.load(f)
    fields = solve(text)
    for k, v in fields.items():
        print(k, v)