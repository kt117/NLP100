def solve(f):
    return f.read().replace('\t', ' ')


with open("chapter02/data/hightemp.txt") as f:
    print(solve(f))