def solve(f):
    lines = [line.split('\t') for line in f.readlines()]
    
    cnt = dict()
    for line in lines:
        cnt[line[0]] = cnt.get(line[0], 0)+1
    cnt = [[v, k] for k, v in cnt.items()]
    return sorted(cnt, reverse=True)


with open("chapter02/data/hightemp.txt") as f:
    print(solve(f))