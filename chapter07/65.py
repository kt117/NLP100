from gensim.models import KeyedVectors
from tqdm import tqdm


with open("chapter07/outputs/64.txt") as f:
    count = {"semantic": [0, 0], "syntactic": [0, 0]}
    current_task = None
    for line in tqdm(f.readlines()):
        words = line[: -1].split()
        if words[0] == ":":
            current_task = "syntactic" if "gram" == words[1][: 4] else "semantic"
        else:
            count[current_task][0] += 1  
            if words[3] == words[4]:
                count[current_task][1] += 1  
    
    for task, c in count.items():
        print(task, c[1] / c[0])
