import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

# Create data
random_state = 21
X, y = make_moons(300, noise=.08)

# Explore data
fig, ax = plt.subplots(figsize=(9,5))
ax.set_title('Half moon data')
ax.scatter(X[:, 0], X[:, 1],s=30)
plt.show()

# Train the model
model = SpectralClustering(
 n_clusters = 2,
 affinity = 'nearest_neighbors',
 n_neighbors = 15,
 assign_labels = 'kmeans')

# Test the model
tags = model.fit_predict(X)

# Draw it
fig, ax = plt.subplots(figsize = (9,5))
ax.set_title('Spectral Clustering')
plt.scatter(X[:, 0], X[:, 1], c = tags, s = 40, cmap = 'plasma')
plt.show()

# Compare with KMeans
modelKmeans = KMeans(n_clusters = 2).fit(X)
tagsKmeans = modelKmeans.predict(X)
plt.title('K - Means')
plt.scatter(X[:, 0], X[:, 1], c = tagsKmeans, s = 40, cmap = 'plasma')
plt.show()