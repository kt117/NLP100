from matplotlib import pyplot as plt
import japanize_matplotlib
import numpy as np
import pickle


def count(morphemes):
    word_to_counts = dict()
    for morpheme in morphemes:
        w = (morpheme["base"], morpheme["pos"], morpheme["pos1"])
        word_to_counts[w] = word_to_counts.get(w, 0) + 1
    return sorted([(w, cnt) for w, cnt in word_to_counts.items()], key=lambda x : x[1], reverse=True)


def draw(word_to_counts):
    plt.bar(
        np.array([i for i in range(len(word_to_counts))]),
        np.array([w[1] for w in word_to_counts]),
        tick_label=[w[0][0] for w in word_to_counts],
    )
    plt.savefig("output/36.png")


def solve(morphemes):
    word_to_counts = count(morphemes)
    draw(word_to_counts[: 10])


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    solve(morphemes)