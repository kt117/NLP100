import sys


def solve(f, n):
    for line in f.readlines()[: n]:
        print(line[: -1])
    return


args = sys.argv
with open("chapter02/data/hightemp.txt") as f:
    solve(f, int(args[1]))