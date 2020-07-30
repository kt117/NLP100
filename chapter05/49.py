from collections import deque
from common_tool import create_chunks_list


def pos_in_chunk(pos, chunk):
    return [morph for morph in chunk.morphs if morph.pos == pos]


def create_paths_list(chunks_list):
    paths_list = list()

    for chunks in chunks_list:
        paths = list()
        
        for i, chunk in enumerate(chunks):
            if len(pos_in_chunk("名詞", chunk)) > 0:
                path = list()

                at = i
                while at != -1:
                    path.append(at)
                    at = chunks[at].dst

                paths.append(path)

        paths_list.append({"chunks": chunks, "paths": paths})

    return paths_list
    

def replace_noun_phrase(chunk, s):
    res = ""
    state = 0

    for morph in chunk.morphs:
        if morph.pos == "名詞" and state == 0:
            state = 1

        if morph.pos != "記号":
            if state != 1:
                res += morph.surface

        if morph.pos == "名詞" and state == 1:
            state = 2
            res += s

    return res


def parse_simple_path(chunks, path):
    path_surfaces = list()

    for i, index in enumerate(path):
        if i == 0:
            path_surfaces.append(replace_noun_phrase(chunks[index], "X"))
        elif i == len(path) - 1:
            path_surfaces.append(replace_noun_phrase(chunks[index], "Y"))
        else:
            path_surfaces.append(str(chunks[index]))

    return " -> ".join(path_surfaces)


def parse_forked_path(chunks, path_a, path_b, lca):
    sa = " -> ".join([replace_noun_phrase(chunks[index], "X") if i == 0 else str(chunks[index]) for i, index in enumerate(path_a)])
    sb = " -> ".join([replace_noun_phrase(chunks[index], "Y") if i == 0 else str(chunks[index]) for i, index in enumerate(path_b)])
    return " | ".join([sa, sb, str(chunks[lca])])


def create_noun_phrase_pair_list(paths_list):
    noun_phrase_pair_list = list()
    
    for chunks_and_paths in paths_list:
        chunks = chunks_and_paths["chunks"]
        paths = chunks_and_paths["paths"]
    
        for i in range(len(paths)):
            for j in range(i):
                path_a = deque(paths[j])
                path_b = deque(paths[i])

                before = -1
                while len(path_a) > 0 and len(path_b) > 0 and path_a[-1] == path_b[-1]:
                    before = path_a.pop()
                    path_b.pop()
                
                if len(path_a) == 0:
                    path_b.append(before)
                    noun_phrase_pair_list.append(parse_simple_path(chunks, path_b))
                elif len(path_b) == 0:
                    path_a.append(before)
                    noun_phrase_pair_list.append(parse_simple_path(chunks, path_a))
                else:
                    noun_phrase_pair_list.append(parse_forked_path(chunks, path_a, path_b, before))

    return noun_phrase_pair_list


def solve(f):
    chunks_list = create_chunks_list(f)
    paths_list = create_paths_list(chunks_list)
    return create_noun_phrase_pair_list(paths_list[: 1])


with open("output/ai.ja.txt.parsed") as f:
    noun_phrase_pair_list = solve(f)
    for s in noun_phrase_pair_list[: 100]:
        print(s)