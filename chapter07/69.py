from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


model = KeyedVectors.load_word2vec_format("chapter07/models/GoogleNews-vectors-negative300.bin", binary=True)

# https://qiita.com/tao_s/items/32b90a2751bfbdd585ea
with open("chapter07/data/country_name.csv") as f:
    df = pd.read_csv(f)

countries = [country.replace(' ', '_') for country in df["ISO 3166-1に於ける英語名"]]
countries = [country for country in countries if country in model.vocab]
vectors = [model[country] for country in countries]
tsne = TSNE(random_state=42).fit_transform(vectors)

plt.figure(figsize=(10, 10))
sns.scatterplot(x=tsne.T[0], y=tsne.T[1])
for i, country in enumerate(countries):
    plt.annotate(country, xy=(tsne[i, 0], tsne[i, 1]))
plt.savefig("chapter07/outputs/69.png")
