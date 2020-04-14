import sys
import MeCab


def solve(f):
    tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

    words = list()
    for line in f.readlines():
        node = tagger.parseToNode(line[: -1])

        while node:
            res = node.feature.split(",")
            word = {"surface": node.surface, "base": res[6], "pos": res[0], "pos1": res[1]}
            words.append(word)
            
            node = node.next

    return words


with open("data/neko.txt") as f:
    words = solve(f)
    for w in words[: 10]:
        print(w)