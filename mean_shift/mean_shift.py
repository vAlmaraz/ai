import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift

# Read data
iris_data = load_iris()
iris = load_iris()
irisData = pd.DataFrame(data = iris.data)
irisData['Species'] = iris.target

# Explore data
# We will only use Petal length and width
plt.scatter(irisData[[2]], irisData[[3]], c = irisData['Species'])
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()

#exit()

# Train and test the model
petals = irisData[[2, 3]].values
model = MeanShift(bandwidth = 0.7).fit(petals)
tags = model.labels_
irisData['tags'] = tags

# Explore model
plt.scatter(irisData[[2]], irisData[[3]], c = irisData['tags'])
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()

#exit()

# Get model accuracy
MC = pd.crosstab(irisData['tags'], irisData['Species'])
accuracy = sum(MC.values.diagonal()) / MC.values.sum()
print("Model accuracy:", accuracy)

# Draw it
centroids = model.cluster_centers_
plt.scatter(irisData[[2]], irisData[[3]], c = irisData['tags'])
plt.scatter(centroids[:, 0], centroids[:, 1], c = ['red', 'green', 'blue'], marker = 'o', s = 100)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()
