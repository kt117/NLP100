import MeCab


def solve(f):
    tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

    morphemes = list()
    for line in f.readlines():
        node = tagger.parseToNode(line[: -1])

        while node:
            res = node.feature.split(",")
            morpheme = {"surface": node.surface, "base": res[6], "pos": res[0], "pos1": res[1]}
            morphemes.append(morpheme)
            
            node = node.next

    return morphemes


with open("data/neko.txt") as f:
    morphemes = solve(f)
    for m in morphemes[: 10]:
        print(m)