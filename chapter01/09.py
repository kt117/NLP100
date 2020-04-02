import random

def solve(s):
    words = s.split(' ')
    words = [w if len(w) <= 4 else w[0]+''.join(random.sample(w[1 : -1], len(w)-2))+w[-1] for w in words]
    return ' '.join(words)

print(solve("I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."))