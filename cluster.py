import pandas

# k-means clustering
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans


df = pandas.read_csv(
    "~/repos/mobile-env/mobile_env/scenarios/very_large/site-optus-melbCBD.csv"
).iloc[1:, 1:3]

df = pandas.DataFrame(df).to_numpy()

model = KMeans(n_clusters=13)
model.fit(df)
yhat = model.predict(df)
print(yhat)
clusters = unique(yhat)


for cluster in clusters:
    row_ix = where(yhat == cluster)
    print(row_ix)

print((model.cluster_centers_[0, 0], model.cluster_centers_[0, 1]))
