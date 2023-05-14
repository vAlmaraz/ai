import numpy as np
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Read data
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

np.random.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_index = np.random.choice(range(X.shape[0]), size=int(X.shape[0]*0.7), replace=False)
X_train = X[train_index, :]
X_test = X[np.setdiff1d(range(X.shape[0]), train_index), :]
y_train = y[train_index]
y_test = y[np.setdiff1d(range(X.shape[0]), train_index)]

n_neighbors = 5
h = .02
cmap_light = ListedColormap(['#FF1AAA', '#AAFAA1', '#AAAAA1'])
cmap_bold = ListedColormap(['#FF0000', '#19FA05', '#AAAA99'])

for weights in ['uniform', 'distance']:
    # Train the model
    model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Get model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy on test data for:", weights, accuracy)

    # Draw it
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
plt.show()