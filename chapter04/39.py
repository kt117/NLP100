from matplotlib import pyplot as plt
import numpy as np
import pickle


def flatten(c):
    return [a for b in c for a in b]


def count(morphemes):
    tfs = dict()
    nwords = 0
    for morpheme in morphemes:
        if morpheme["pos"] != "BOS/EOS": 
            w = (morpheme["base"], morpheme["pos"], morpheme["pos1"])
            tfs[w] = tfs.get(w, 0) + 1
            nwords += 1
    return sorted([(w, cnt / nwords) for w, cnt in tfs.items()], key=lambda x : x[1], reverse=True)


def draw(tfs):
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(
        np.array([i + 1 for i in range(len(tfs))]),
        np.array([w[1] for w in tfs])
    )
    plt.savefig("output/39.png")


def solve(morphemes):
    morphemes = flatten(morphemes)
    tfs = count(morphemes)
    draw(tfs)


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    solve(morphemes)