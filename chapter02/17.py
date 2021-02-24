def solve(f):
    return {line.split('\t')[0] for line in f.readlines()}


with open("chapter02/data/hightemp.txt") as f:
    print(solve(f))