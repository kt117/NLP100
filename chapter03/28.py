import pickle
import re


class MarkupRemover:
    extract_patterns = [
        r"(\'{2,5})(.*)(\1)",  # 強調
        r"\[\[[^\[\|]*?\|?([^\[\|]*)\]\]",  # 内部リンク
        r"\[\[ファイル:.+\|.+\|(.*)\]\]",  # file
        r"\{\{lang\|.+\|(.+)\}\}",  # lang
    ]
    extract_patterns = [re.compile(p) for p in extract_patterns]
    poss = [r"\2", r"\1", r"\1", r"\1"]

    remove_patterns = [
        r"\[http://.+\]",  # link
        r"<ref.*>",  # ref
        r"<br */>",  # br
    ]
    remove_patterns = [re.compile(p) for p in remove_patterns]

    def remove(self, s):
        for p, pos in zip(self.extract_patterns, self.poss):
            s = re.sub(p, pos, s)
        for p in self.remove_patterns:
            s = re.sub(p, "", s)
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


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    fields = solve(text)
    for k, v in fields.items():
        print(k, v)