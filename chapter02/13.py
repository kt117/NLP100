def solve(f, g):
    flines = f.readlines()
    glines = g.readlines()
    hlines = [fl[: -1] + '\t' + gl[: -1] + '\n' for fl, gl in zip(flines, glines)]
    return ''.join(hlines)


with open("output/col1.txt") as f0, open("output/col2.txt") as f1, open("output/res13.txt", mode='w') as h:
    h.write(solve(f0, f1))