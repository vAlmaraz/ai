import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read Weather data file
weatherData = pd.read_csv('weather.csv')

# Ignore dates (in order to prevent error when predicting)
weatherData = weatherData.drop('Date', axis=1)

# Omit rows without available values
weatherData = weatherData.dropna()
# Reset indexes after removing rows
weatherData = weatherData.reset_index(drop=True)

# Convert RainToday and RainTomorrow into 0 and 1 instead of No and Yes
weatherData["RainToday"] = np.where(weatherData["RainToday"] == "Yes", 1, 0)
weatherData["RainTomorrow"] = np.where(weatherData["RainTomorrow"] == "Yes", 1, 0)

X = weatherData.iloc[:, :-1]
y = weatherData.iloc[:, -1]

np.random.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
indexes = np.random.permutation(len(weatherData))
train_size = int(len(weatherData) * 0.7)
train_idx, test_idx = indexes[:train_size], indexes[train_size:]
# Split data into train and test
X_train, y_train = X.iloc[train_idx, :], y[train_idx]
X_test, y_test = X.iloc[test_idx, :], y[test_idx]

# Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might end up creating a bias
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Get model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test data:", accuracy)

# Draw it
# TODO: Draw it