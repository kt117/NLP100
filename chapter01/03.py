def solve(s):
    s = s.replace(',', '').replace('.', '')
    return [len(w) for w in s.split(' ')]

print(solve("Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."))