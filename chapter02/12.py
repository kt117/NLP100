def solve(f):
    res = ["", ""]
    for line in f.readlines():
        line = line.split('\t')
        res[0] += line[0] + '\n'
        res[1] += line[1] + '\n'
    return res[0], res[1]


with open("data/hightemp.txt") as f:
    res = solve(f)

    with open("output/col1.txt", mode='w') as f0:
        f0.write(res[0])

    with open("output/col2.txt", mode='w') as f1:
        f1.write(res[1])