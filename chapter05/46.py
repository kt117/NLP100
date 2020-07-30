from common_tool import create_chunks_list


def pos_in_chunk(pos, chunk):
    return [morph for morph in chunk.morphs if morph.pos == pos]


def solve(f):
    chunks_all = create_chunks_list(f)
    verb_to_cases_list = list()

    for chunks in chunks_all:
        chunk_to_cases = [list() for _ in chunks] 
        chunk_to_chunks = [list() for _ in chunks] 

        for chunk in chunks:
            if chunk.dst != -1:
                cases = pos_in_chunk("助詞", chunk)
                if len(cases) > 0:
                    chunk_to_cases[chunk.dst] += cases
                    chunk_to_chunks[chunk.dst].append(chunk)

        for i, cases in enumerate(chunk_to_cases):
            verbs = pos_in_chunk("動詞", chunks[i])

            if len(verbs) > 0 and len(cases) > 0:
                verb_to_cases_list.append({"verb": verbs[0], "cases": cases, "chunks": chunk_to_chunks[i]})

    return verb_to_cases_list


with open("output/ai.ja.txt.parsed") as f:
    verb_to_cases_list = solve(f)

    for verb_to_cases in verb_to_cases_list[: 10]:
        case_surfaces = [case.surface for case in verb_to_cases["cases"]]
        chunk_surfaces = [str(chunk) for chunk in verb_to_cases["chunks"]]
        print(verb_to_cases["verb"].base + "\t" + '\t'.join(case_surfaces) + "\t" + '\t'.join(chunk_surfaces))