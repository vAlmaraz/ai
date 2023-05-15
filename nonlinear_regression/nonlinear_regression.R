library(tidyverse)
library(caret)

# Read data
X <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
y <- c(1, 3, 9, 15, 25, 36, 50, 65, 81, 100, 121, 143, 169, 196, 225, 256, 289, 324, 361, 400, 441)

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
split <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[split]
X_test <- X[-split]
y_train <- y[split]
y_test <- y[-split]

# Train the model
model <- nls(y_train ~ a * X_train^b, start = list(a = 1, b = 1))

# Test the model
y_pred <- predict(model, newdata = data.frame(X_train = X_test))

# Get model accuracy
accuracy <- cor.test(y_pred, y_test)$r.squared
print(paste("Model accuracy on test data:", accuracy))

# Draw it
plot(X_test, y_test, col = "blue", xlab = "x", ylab = "y", main = "Test data")
lines(X_test, y_pred, col = "red", lwd = 2, type = "l")
legend("topright", legend = c("Test data", "Nonlinear regression"), col = c("blue", "red"), lwd = c(1, 2))