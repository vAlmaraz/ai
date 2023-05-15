library(ggplot2)
library(caret)
library(dplyr)

# Read data
salaryData <- read.csv('../datasets/Salary_Data.csv')

X <- salaryData[, 1]
y <- salaryData[, 2]

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_indexes <- createDataPartition(y, times = 1, p = 0.7, list = FALSE)
X_train <- X[train_indexes]
X_test <- X[-train_indexes]
y_train <- y[train_indexes]
y_test <- y[-train_indexes]

# Train the model
model <- lm(y_train ~ X_train, salaryData)
print(coefficients(model))

# Test the model
y_pred <- predict(model, newdata = data.frame(X_train = X_test))

# Get model accuracy
mse <- mean((y_test - y_pred)^2)
print(paste("Mean Squared Error:", mse))
r2 <- as.numeric(cor(y_test, y_pred)^2)
print(paste("R-squared:", r2))

# Draw it
plot(y_train ~ X_train, data = salaryData)
abline(model)

ggplot() +
  geom_point(data = data.frame(X_test, y_test), aes(x = X_test, y = y_test), color = "lightblue") +
  geom_line(data = data.frame(X_test, y_test), aes(x = X_test, y = y_test), color = "blue") +
  geom_point(data = data.frame(X_test, y_pred), aes(x = X_test, y = y_pred), color = "lightgreen") +
  geom_line(data = data.frame(X_test, y_pred), aes(x = X_test, y = y_pred), color = "green") +
  labs(title = "Salary VS Experience", x = "Experience (years)", y = "Salary")
