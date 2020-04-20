import pickle


def solve(morphemes):
    return [morpheme["surface"] for morpheme in morphemes if morpheme["pos"] == "動詞"]


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    surfaces = solve(morphemes)
    for surface in surfaces[: 10]:
        print(surface)