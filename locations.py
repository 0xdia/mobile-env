import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
import matplotlib.cm as cm

df = pd.read_csv(
    "~/repos/mobile-env/mobile_env/scenarios/very_large/site-optus-melbCBD.csv"
).iloc[:, 1:3]

colors = cm.rainbow(np.linspace(0, 1, 13))

fig, ax = plt.subplots()

newdf = pd.DataFrame(df).to_numpy()
model = KMeans(n_clusters=13)
model.fit(newdf)
yhat = model.predict(newdf)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    ax.scatter(newdf[row_ix, 1], newdf[row_ix, 0], color=colors[cluster])

ax.scatter(model.cluster_centers_[:, 1], model.cluster_centers_[:, 0], color="black")

radius = 0.003  # Adjust the radius of the circles as needed
for xi, yi in zip(model.cluster_centers_[:, 1], model.cluster_centers_[:, 0]):
    circle = Circle((xi, yi), radius, edgecolor="black", facecolor="none")
    ax.add_patch(circle)
ax.set_aspect("equal")


ax.set_title("Edge servers, Base stations")
ax.set_xlim(
    min(model.cluster_centers_[:, 1].min(), df.LONGITUDE.min()),
    max(model.cluster_centers_[:, 1].max(), df.LONGITUDE.max()),
)
ax.set_ylim(
    min(model.cluster_centers_[:, 0].min(), df.LATITUDE.min()),
    max(model.cluster_centers_[:, 0].max(), df.LATITUDE.max()),
)


plt.show()
