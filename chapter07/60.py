from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)
print(model["United_States"])