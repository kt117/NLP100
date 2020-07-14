def encode(s):
    res = ""
    for c in s:
        res += chr(219 - ord(c)) if (ord('a') <= ord(c) <= ord('z')) else c
    return res


def decode(s):
    res = ""
    for c in s:
        res += chr(219 - ord(c)) if (ord('a') <= 219 - ord(c) <= ord('z')) else c
    return res


s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
print(encode(s))
print(decode(encode(s)))