def solve(f):
    return len(f.readlines())


with open("chapter02/data/hightemp.txt") as f:
    print(solve(f))