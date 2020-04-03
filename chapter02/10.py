def solve(f):
    return len(f.readlines())


with open("data/hightemp.txt") as f:
    print(solve(f))