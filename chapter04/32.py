import pickle


def flatten(c):
    return [a for b in c for a in b]


def solve(morphemes):
    morphemes = flatten(morphemes)
    return [morpheme["base"] for morpheme in morphemes if morpheme["pos"] == "動詞"]


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    bases = solve(morphemes)
    for base in bases[: 10]:
        print(base)