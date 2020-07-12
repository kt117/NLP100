import CaboCha


def read_file(f):
    res = list()
    for line in f.readlines():
        res.append(line[: -1])
    return res[2 :]


def cabocha_parse(f, g):
    sentences = read_file(f)

    parser = CaboCha.Parser()
    for sentence in sentences:
        g.write(parser.parse(sentence).toString(CaboCha.FORMAT_LATTICE))


with open("data/ai.ja.txt") as f, open("output/ai.ja.txt.parsed", mode='w') as g:
    cabocha_parse(f, g)
    

class Morph:
    def __init__(self, morpheme):
        self.surface = morpheme["surface"]
        self.base = morpheme["base"]
        self.pos = morpheme["pos"]
        self.pos1 = morpheme["pos1"]


def solve(f):
    morphemes_all = list()

    morphemes = list()
    for line in f.readlines():
        if line[0] == '*':
            continue

        if line == "EOS\n":
            morphemes_all.append(morphemes)
            morphemes = list()
        else:
            morpheme = line[: -1].split('\t')
            surface = morpheme[0]
            morpheme = morpheme[1].split(',')
            morphemes.append(Morph({"surface": surface, "base": morpheme[6], "pos": morpheme[0], "pos1": morpheme[1]}))

    return morphemes_all


with open("output/ai.ja.txt.parsed") as f:
    morphemes_all = solve(f)

    for morphemes in morphemes_all[: 10]:
        for morpheme in morphemes:
            print(morpheme.surface)
        print("")