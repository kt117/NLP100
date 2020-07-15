import pydot


class Morph:
    def __init__(self, morpheme):
        self.surface = morpheme["surface"]
        self.base = morpheme["base"]
        self.pos = morpheme["pos"]
        self.pos1 = morpheme["pos1"]


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


def draw(edges):
    graph = pydot.graph_from_edges(edges)
    graph.write_png("output/44.png", prog="dot")


def solve(f):
    chunks_all = create_chunks_list(f)

    for chunks in chunks_all[: 1]:
        edges = list()

        for chunk in chunks:
            if chunk.dst != -1:
                edges.append([str(chunk), str(chunks[chunk.dst])])

        draw(edges)


with open("output/ai.ja.txt.parsed") as f:
    solve(f)