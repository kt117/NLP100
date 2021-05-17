from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import pandas as pd


model = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)

# https://qiita.com/tao_s/items/32b90a2751bfbdd585ea
with open("chapter07/data/country_name.csv") as f:
    df = pd.read_csv(f)

K = 5
countries = [country.replace(' ', '_') for country in df["ISO 3166-1に於ける英語名"]]
countries = [country for country in countries if country in model.vocab]
vectors = [model[country] for country in countries]
clusters = KMeans(n_clusters=K, random_state=42).fit_predict(vectors)

countries_clustered = [list() for _ in  range(K)]
for country, cluster in zip(countries, clusters):
    countries_clustered[cluster].append(country)

for i in range(K):
    print(f"cluster No.{i}")
    print(countries_clustered[i])
