from common_tool import create_chunks_list


def pos_in_chunk(pos, chunk):
    return [morph for morph in chunk.morphs if morph.pos == pos]


def solve(f):
    chunks_all = create_chunks_list(f)
    paths = list()

    for chunks in chunks_all:
        for i, chunk in enumerate(chunks):
            if len(pos_in_chunk("名詞", chunk)) > 0:
                path = list()

                at = i
                while at != -1:
                    path.append(str(chunks[at]))
                    at = chunks[at].dst

                paths.append(path)

    return paths


with open("output/ai.ja.txt.parsed") as f:
    paths = solve(f)

    for path in paths[: 10]:
        print(" -> ".join(path))