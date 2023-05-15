from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read data
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

# Train model
k = 2
kmeans = KMeans(n_clusters = k, n_init = 'auto')
kmeans.fit(irisData[[2, 3]])
irisData['cluster'] = kmeans.labels_

# Explore model
plt.scatter(irisData[[2]], irisData[[3]], c = irisData['cluster'])
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()

#exit()

# Setosa correctly isolated. Virginica and versicolor mixed
# Create a table and elbow function to explore the results from k 1 to 10
cluster_table = pd.crosstab(irisData['cluster'], irisData['Species'])
print(cluster_table)
#exit()

def elbow_fun(df):
    num_k = []
    error = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df)
        num_k.append(i)
        error.append(kmeans.inertia_)
    df_new = pd.DataFrame({'Num_k': num_k, 'Error': error})
    return df_new

df = irisData[[2, 3]]
elbow = elbow_fun(df)

# Draw elbow function chart
plt.plot(elbow['Num_k'], elbow['Error'])
plt.scatter(elbow['Num_k'], elbow['Error'], color='blue', s=30)
plt.xlabel("Num_k")
plt.ylabel("Error")
plt.show()

#exit()

# Conclusion: drastical error reduction using 2, and still reducing with 3.
# No variation higher than 3 due to minimal variance

# Apply k = 3
k = 3
kmeans = KMeans(n_clusters = k, n_init = 'auto')
kmeans.fit(irisData[[2, 3]])
irisData['cluster'] = kmeans.labels_

# Draw it
plt.scatter(irisData[[2]], irisData[[3]], c = irisData['cluster'])
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()