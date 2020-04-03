def bi_gram(s):
    return {s[i : i+2] for i in range(len(s)-1)}


def solve(s, t):
    X = bi_gram(s)
    Y = bi_gram(t)
    return X, Y, X | Y, X & Y, X - Y, "se" in X, "se" in Y


print(solve("paraparaparadise", "paragraph"))