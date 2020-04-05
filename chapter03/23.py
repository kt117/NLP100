import pickle
import re


def solve(text):
    pattern = re.compile(r'=+.+=+$')
    res = list()
    for line in text.split('\n'):
        if re.match(pattern, line):
            level = 0
            while(line[0] == '='):
                line = line[1 : -1]
                level += 1
            res.append([line.strip(), level-1])
    return res


with open("output/england.pickle", "rb") as f:
    text = pickle.load(f)
    print(solve(text))