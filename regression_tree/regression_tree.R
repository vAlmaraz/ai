library(ggplot2)
library(dplyr)
library(caret)
library(e1071)
library(scales)

# Read data
dataset <- read.csv('../datasets/Position_Salaries.csv')

X <- dataset[, 2]
y <- dataset[, 3]

# Train the model
regressor <- svm(X, y)

# Test the model
y_pred <- predict(regressor, 6.5)
print(y_pred)

# Draw it
X_grid <- seq(min(X), max(X), by = 0.01)
X_grid <- data.frame(X_grid)
names(X_grid) <- "X"
ggplot() +
  geom_point(aes(x = X, y = y), data = dataset, color = "red") +
  geom_line(aes(x = X_grid$X, y = predict(regressor, X_grid)), color = "blue") +
  ggtitle("Regression Tree") +
  xlab("Position") +
  ylab("Salary")
