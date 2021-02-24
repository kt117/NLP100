def solve(f, g):
    flines = f.readlines()
    glines = g.readlines()
    hlines = [fl[: -1] + '\t' + gl[: -1] + '\n' for fl, gl in zip(flines, glines)]
    return ''.join(hlines)


with open("chapter02/outputs/col1.txt") as f0, open("chapter02/outputs/col2.txt") as f1, open("chapter02/outputs/res13.txt", mode='w') as h:
    h.write(solve(f0, f1))