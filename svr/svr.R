library(e1071)
library(scales)
library(ggplot2)

# Read data
dataset <- read.csv('../datasets/Position_Salaries.csv')

X <- dataset[, 2]
y <- dataset[, 3]

# Train the model
model <- svm(y ~ X, kernel = 'radial', cost = 1.0, epsilon = 0.2)

# Test the model
y_pred <- predict(model, newdata = data.frame(X = c(6.5)))

# Draw it
x_grid <- seq(min(X), max(X), by = 0.01)
y_grid <- predict(model, newdata = data.frame(X = x_grid))

data <- data.frame(X = as.vector(X), y = as.vector(y))
grid <- data.frame(X = as.vector(x_grid), y = as.vector(y_grid))

ggplot() +
  geom_point(data = data, aes(x = X, y = y), color = 'red') +
  geom_line(data = grid, aes(x = X, y = y), color = 'blue') +
  ggtitle('SVR Kernel = RBF') +
  xlab('Years Experience') +
  ylab('Salary')