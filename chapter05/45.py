class Morph:
    def __init__(self, morpheme):
        self.surface = morpheme["surface"]
        self.base = morpheme["base"]
        self.pos = morpheme["pos"]
        self.pos1 = morpheme["pos1"]

    def __str__(self):
        return self.surface


class Chunk:
    def __init__(self, dst):
        self.morphs = list()
        self.dst = dst
        self.srcs = list()
        
    def __str__(self):
        return ''.join([morph.surface for morph in self.morphs if morph.pos != "記号"])


def create_chunks_list(f):
    chunks_all = list()

    chunk = None
    chunks = list()
    for line in f.readlines():
        if line[0] == '*':
            if chunk is not None:
                chunks.append(chunk)
            chunk = Chunk(int(line.split(' ')[2][: -1]))
            continue

        if line == "EOS\n":
            if chunk is not None:
                chunks.append(chunk)
            chunk = None

            for i, c in enumerate(chunks):
                chunks[c.dst].srcs.append(i)
            
            if len(chunks) > 0:
                chunks_all.append(chunks)
                chunks = list()
        else:
            morpheme = line[: -1].split('\t')
            surface = morpheme[0]
            morpheme = morpheme[1].split(',')
            chunk.morphs.append(Morph({"surface": surface, "base": morpheme[6], "pos": morpheme[0], "pos1": morpheme[1]}))

    return chunks_all


def pos_in_chunk(pos, chunk):
    return [morph for morph in chunk.morphs if morph.pos == pos]


def solve(f):
    chunks_all = create_chunks_list(f)

    verb_to_cases_list = list()

    for chunks in chunks_all:
        chunk_to_cases = [list() for _ in chunks] 

        for chunk in chunks:
            if chunk.dst != -1:
                chunk_to_cases[chunk.dst] += pos_in_chunk("助詞", chunk)

        for i, cases in enumerate(chunk_to_cases):
            verbs = pos_in_chunk("動詞", chunks[i])

            if len(verbs) > 0 and len(cases) > 0:
                verb_to_cases_list.append({"verb": verbs[0], "cases": cases})

    return verb_to_cases_list


with open("output/ai.ja.txt.parsed") as f:
    verb_to_cases_list = solve(f)


with open("output/45.txt", mode='w') as g:
    for verb_to_cases in verb_to_cases_list:
        case_surfaces = [case.surface for case in verb_to_cases["cases"]]
        g.write(verb_to_cases["verb"].base + " " + ' '.join(case_surfaces) + "\n")