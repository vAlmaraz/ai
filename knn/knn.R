library(class)
library(datasets)

# Read data
iris <- datasets::iris

X <- iris[, 1:4]
y <- iris[, 5]

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_index <- sample(nrow(X), size = round(nrow(X)*0.7), replace = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

n_neighbors <- 5
# Train & test the model
y_pred <- knn(X_train, X_test, y_train, k = n_neighbors)

# Get model accuracy
accuracy <- mean(y_pred == y_test)
print(paste0("Model accuracy on test data: ", accuracy))

# Draw it
data <- cbind(X_test, y_pred)
colnames(data) <- c(colnames(X_test), "Species")
plot(X_test[, 1], X_test[, 2], col = y_pred, pch = 19, xlab = "x1", ylab = "x2")
points(X_test[, 1], X_test[, 2], col = y_test, pch = 4)
