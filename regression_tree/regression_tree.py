import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Read data
dataset = pd.read_csv('../datasets/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Train the model
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Test the model
x_trans = sc_X.transform([[6.5]])
y_pred = regressor.predict(x_trans)
y_pred = sc_y.inverse_transform(y_pred.reshape(1, -1))
print(y_pred)

# Draw it
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Regression Tree')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()