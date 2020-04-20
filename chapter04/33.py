import pickle

def solve(morphemes):
    res = list()
    for i in range(len(morphemes)-2):
        if morphemes[i]["pos"] == "名詞" and morphemes[i + 1]["surface"] == "の" and morphemes[i + 2]["pos"] == "名詞":
            res.append(morphemes[i]["surface"] + morphemes[i + 1]["surface"] + morphemes[i + 2]["surface"]) 
    return res


with open("output/morphemes.pickle", mode="rb") as f:
    morphemes = pickle.load(f)
    anobs = solve(morphemes)
    for anob in anobs[: 10]:
        print(anob)