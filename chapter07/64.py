from gensim.models import KeyedVectors
from tqdm import tqdm


model = KeyedVectors.load_word2vec_format('chapter07/data/GoogleNews-vectors-negative300.bin', binary=True)

with open("chapter07/data/questions-words.txt") as f, open("chapter07/outputs/64.txt", 'w') as g:
    for line in tqdm(f.readlines()):
        words = line[: -1].split()
        if words[0] != ':':
            word, similarity = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
            words += [word, str(similarity)]
        g.write(' '.join(words) + '\n')