import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Read Cats data file
catsData = pd.read_csv('dataset-32337.csv')

# Convert Sex into 0 and 1 instead of M and F
catsData["Sex"] = np.where(catsData["Sex"] == "F", 1, 0)

X = catsData.iloc[:, 1:]
y = catsData.iloc[:, 0]

np.random.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
indexes = np.random.permutation(len(catsData))
train_size = int(len(catsData) * 0.7)
train_idx, test_idx = indexes[:train_size], indexes[train_size:]
# Split data into train and test
X_train, y_train = X.iloc[train_idx, :], y[train_idx]
X_test, y_test = X.iloc[test_idx, :], y[test_idx]

# Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might end up creating a bias
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Train the model
C = 1.0
modelLinear = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
modelLinear2 = svm.LinearSVC(C=C).fit(X_train, y_train)
modelRbf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
modelPoly = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
# Possible error:
#     raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous'
# Why? It's detecting as regression but forced as classification. As we are trying to classify, something is wrong with the data
# Solution: Review X and y

# Possible error:
# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
# Why? Normally when an optimization algorithm does not converge, it is usually because the problem is not well-conditioned, perhaps due to a poor scaling of the decision variables
# Solution: scale data

# Kernel:
# precomputed, linear, poly, rbf, sigmoid

# Test the model
y_predLinear = modelLinear.predict(X_test)
y_predLinear2 = modelLinear2.predict(X_test)
y_predRbf = modelRbf.predict(X_test)
y_predPoly = modelPoly.predict(X_test)

# Get model accuracy
print("Linear model accuracy:", modelLinear.score(X_test, y_test))
print("Linear 2 model accuracy:", modelLinear2.score(X_test, y_test))
print("Rbf model accuracy:", modelRbf.score(X_test, y_test))
print("Poly model accuracy:", modelPoly.score(X_test, y_test))

# Draw it
h = 0.2
# Plots frames
x_min, x_max = X["Bwt"].min() - 1, X["Bwt"].max() + 1
y_min, y_max = X["Hwt"].min() - 1, X["Hwt"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

titles = ['SVM (kernel=lineal)', 'Lineal SVM', 'SVM (kernel=RBF)', 'SVM (kernel=degree 3 polynomial)']

for i, clf in enumerate((modelLinear, modelLinear2, modelRbf, modelPoly)):
    # Limits between plots
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Chart color
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    # Train model
    plt.scatter(X["Bwt"], X["Hwt"], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal weight')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()