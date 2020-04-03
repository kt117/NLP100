def solve(s):
    first = {1, 5, 6, 7, 8, 9, 15, 16, 19}
    s = s.replace('.', '').split(' ')
    return [w[0] if i+1 in first else w[: 2] for i, w in enumerate(s)]


print(solve("Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."))