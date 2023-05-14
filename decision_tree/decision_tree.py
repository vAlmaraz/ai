import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz 
import pydotplus
from sklearn.tree import export_graphviz

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
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Get model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test data:", accuracy)

# Draw it
dot_data = export_graphviz(model, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graphviz.Source(graph.to_string())