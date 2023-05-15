import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mtp
from matplotlib.colors import ListedColormap

# Read data
iris = load_iris()

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
model = RandomForestClassifier(n_estimators= 10, criterion="entropy")
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Get model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test data:", accuracy)

# Draw it
feature_importance = model.feature_importances_
mtp.figure(figsize=(8, 6))
mtp.bar(range(len(feature_importance)), feature_importance, tick_label=np.arange(1, len(feature_importance)+1))
mtp.xlabel('Characteristic')
mtp.ylabel('Importance')
mtp.title('Characteristic importance in Random Forest')
mtp.show()