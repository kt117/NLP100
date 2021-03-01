from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format("chapter07/data/GoogleNews-vectors-negative300.bin", binary=True)
print(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"]))
