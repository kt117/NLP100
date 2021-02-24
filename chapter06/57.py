import pickle
import numpy as np


with open('chapter06/models/count_vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

with open('chapter06/models/classifier.pickle', 'rb') as f:
    model = pickle.load(f)

names = vectorizer.get_feature_names()
for coef in model.coef_:
    sorted_index = np.argsort(coef)
    print([names[sorted_index[- i - 1]] for i in range(10)])
    print([names[sorted_index[i]] for i in range(10)])
