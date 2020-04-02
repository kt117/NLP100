def solve(s, t):
    res = ""
    for a, b in zip(s, t):
        res += a+b
    return res

print(solve("パトカー", "タクシー"))