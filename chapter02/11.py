def solve(f):
    return f.read().replace('\t', ' ')


with open("data/hightemp.txt") as f:
    print(solve(f))