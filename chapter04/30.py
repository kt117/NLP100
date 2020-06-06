import MeCab
import pickle


def solve(f):
    tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

    morphemes_all = list()
    for line in f.readlines():
        node = tagger.parseToNode(line[: -1])
        morphemes = list()

        while node:
            res = node.feature.split(",")
            morpheme = {"surface": node.surface, "base": res[6], "pos": res[0], "pos1": res[1]}
            morphemes.append(morpheme)
            
            node = node.next

        morphemes_all.append(morphemes)

    return morphemes_all


with open("data/neko.txt") as f, open("output/morphemes.pickle", mode='wb') as g:
    morphemes_all = solve(f)
    pickle.dump(morphemes_all, g)
    for m in morphemes_all[: 10]:
        print(m)