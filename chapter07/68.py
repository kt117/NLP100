from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd


model = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)

# https://qiita.com/tao_s/items/32b90a2751bfbdd585ea
with open("chapter07/data/country_name.csv") as f:
    df = pd.read_csv(f)

countries = [country.replace(' ', '_') for country in df["ISO 3166-1に於ける英語名"]]
countries = [country for country in countries if country in model.vocab]
vectors = [model[country] for country in countries]
linkage_result = linkage(vectors, method='ward')

plt.figure(figsize=(10, 30))
dendrogram(linkage_result, labels=countries, orientation='right')
plt.savefig("chapter07/outputs/68.png")
