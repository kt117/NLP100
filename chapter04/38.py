from matplotlib import pyplot as plt
import numpy as np
import pickle


def flatten(c):
    return [a for b in c for a in b]


def count(morphemes):
    word_to_counts = dict()
    for morpheme in morphemes:
        if morpheme["pos"] != "BOS/EOS": 
            w = (morpheme["base"], morpheme["pos"], morpheme["pos1"])
            word_to_counts[w] = word_to_counts.get(w, 0) + 1
    return sorted([(w, cnt) for w, cnt in word_to_counts.items()], key=lambda x : x[1], reverse=True)


def draw(word_to_counts):
    plt.hist(
        np.array([w[1] for w in word_to_counts]),
        range=(1, 20),
        bins=20,
    )
    plt.savefig("output/38.png")


def solve(morphemes):
    morphemes = flatten(morphemes)
    word_to_counts = count(morphemes)
    draw(word_to_counts)


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    solve(morphemes)