import pickle


def flatten(c):
    return [a for b in c for a in b]
    

def solve(morphemes):
    morphemes = flatten(morphemes)

    word_to_counts = dict()
    for morpheme in morphemes:
        w = (morpheme["base"], morpheme["pos"], morpheme["pos1"])
        word_to_counts[w] = word_to_counts.get(w, 0) + 1
    return sorted([(w, cnt) for w, cnt in word_to_counts.items()], key=lambda x : x[1], reverse=True)


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    word_to_counts = solve(morphemes)
    for w in word_to_counts[: 20]:
        print(w)