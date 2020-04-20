import pickle


def solve(morphemes):
    res = list()
    s = ""
    cnt = 0
    for morpheme in morphemes:
        if morpheme["pos"] == "名詞":
            s += morpheme["surface"]
            cnt += 1
        else:
            if cnt > 1:
                res.append(s)
            s = ""
            cnt = 0
    if cnt > 1:
        res.append(s)
    return res


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    nouns = solve(morphemes)
    for noun in nouns[: 10]:
        print(noun)