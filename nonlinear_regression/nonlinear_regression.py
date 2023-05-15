import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Read data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
y = np.array([1, 3, 9, 15, 25, 36, 50, 65, 81, 100, 121, 143, 169, 196, 225, 256, 289, 324, 361, 400, 441])

poly_features = PolynomialFeatures(degree = 2)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))

# Calculate random indexes and split into two groups. Train data will include 70% of data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Get model accuracy
accuracy = model.score(X_test, y_test)
print("Model accuracy on test data:", accuracy)

# Draw it
plt.scatter(X_test[:, 1], y_test, label='Datos de prueba')
plt.plot(X_test[:, 1], y_pred, color='red', label='Valores predichos')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()