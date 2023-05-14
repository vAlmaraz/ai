from sklearn.datasets import load_iris
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read data
iris = load_iris()
irisData = iris.data

X = iris.data
y = iris.target

np.random.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_index = np.random.choice(range(X.shape[0]), size=int(X.shape[0]*0.7), replace=False)
X_train = X[train_index, :]
X_test = X[np.setdiff1d(range(X.shape[0]), train_index), :]
y_train = y[train_index]
y_test = y[np.setdiff1d(range(X.shape[0]), train_index)]

# Train the model
model = LDA()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Get model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test data:", accuracy)

# Draw it
plt.subplot(2, 2, 1)
color = np.where(y_test == 0, 'red', np.where(y_test == 1, 'blue', 'green'))
plt.scatter(X_test[:, 0], X_test[:, 1], c=color)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Original')

plt.subplot(2, 2, 2)
color = np.where(y_pred == 0, 'red', np.where(y_pred == 1, 'blue', 'green'))
plt.scatter(X_test[:, 0], X_test[:, 1], c=color)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Predicted')

plt.subplot(2, 2, 3)
color = np.where(y_test == 0, 'red', np.where(y_test == 2, 'blue', 'green'))
plt.scatter(X_test[:, 0], X_test[:, 2], c=color)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Original')

plt.subplot(2, 2, 4)
color = np.where(y_pred == 0, 'red', np.where(y_pred == 2, 'blue', 'green'))
plt.scatter(X_test[:, 0], X_test[:, 2], c=color)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Predicted')

plt.tight_layout()
plt.show()