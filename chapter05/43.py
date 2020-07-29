from common_tool import create_chunks_list


def pos_in_chunk(pos, chunk):
    res = False
    for morph in chunk.morphs:
        if morph.pos == pos:
            res = True
    return res


def solve(f):
    chunks_all = create_chunks_list(f)

    for chunks in chunks_all[: 1]:
        for chunk_a in chunks:
            if chunk_a.dst != -1:
                chunk_b = chunks[chunk_a.dst]
                if pos_in_chunk("名詞", chunk_a) and pos_in_chunk("動詞", chunk_b):
                    print(chunk_a, chunk_b)


with open("output/ai.ja.txt.parsed") as f:
    solve(f)