def solve(s, n):
    words = s.split(' ')
    return [[words[i+j] for j in range(n)] for i in range(len(words)-n+1)], [s[i : i+n] for i in range(len(s)-n+1)]


print(solve("I am an NLPer", 2))