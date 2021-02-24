def solve(f):
    lines = [line.split('\t') for line in f.readlines()]
    lines.sort(key=lambda x: x[2], reverse=True)
    return ''.join(['\t'.join(line) for line in lines])


with open("chapter02/data/hightemp.txt") as f:
    print(solve(f))