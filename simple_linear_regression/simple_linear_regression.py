import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read data
dataset = pd.read_csv('../datasets/Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Calculate random indexes and split into two groups. Train data will include 70% of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Get model accuracy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Draw it
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train),color='blue')
plt.title('Salary VS Experience (Training data)')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, model.predict(X_test),color='blue')
plt.title('Salary VS Experience (Test data)')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()