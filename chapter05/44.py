from common_tool import create_chunks_list
import pydot


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