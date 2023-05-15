import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Read data
dataset = pd.read_csv('../datasets/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Train the model
model = SVR(kernel = 'rbf', C=1.0, epsilon=0.2)
model.fit(X, np.ravel(y))

# Test the model
x_trans = sc_X.transform([[6.5]])
y_pred = model.predict(x_trans)
y_pred = sc_y.inverse_transform(y_pred.reshape(1, -1))

# Get model accuracy

# Draw it
x_real = sc_X.inverse_transform(X)
y_real = sc_y.inverse_transform(y)

X_grid = np.arange(min(x_real), max(x_real), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
x_grid_transform = sc_X.transform(X_grid)
y_grid = model.predict(x_grid_transform)
y_grid = y_grid.reshape(len(y_grid), 1)
y_grid_real = sc_y.inverse_transform(y_grid)

plt.scatter(x_real, y_real, color = 'red')
plt.plot(X_grid, y_grid_real, color = 'blue')
plt.title('SVR Kernel=RBF')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()