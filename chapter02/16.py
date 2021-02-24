import sys


def solve(f, n):
    lines = f.readlines()
    batch_size = (len(lines) + n - 1) // n
    for i in range(n):
        with open(f"chapter02/outputs/res16_{i}.txt", mode='w') as f:
            f.write(''.join(lines[i * batch_size : (i + 1) * batch_size]))
    return


args = sys.argv
with open("chapter02/data/hightemp.txt") as f:
    solve(f, int(args[1]))