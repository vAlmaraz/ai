library(randomForest)
library(ggplot2)

# Read data
iris <- datasets::iris

X <- iris[, 1:4]
y <- iris[, 5]

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_index <- sample(nrow(iris), size = round(0.7*nrow(iris)), replace = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Train the model
model <- randomForest(x = X_train, y = y_train, ntree = 100, importance = TRUE)

# Test the model
predictions <- predict(model, test, type = "class")

# Get model accuracy
accuracy <- sum(predictions == test$Species) / nrow(test)
print(paste0("Model accuracy on test data: ", round(accuracy, 2)))

# Draw it
p <- ncol(X_train)
feature_importance <- model$importance[, "MeanDecreaseAccuracy"]
importance_df <- data.frame(
  Feature = 1:p,
  Importance = feature_importance
)
ggplot(importance_df, aes(x = Feature, y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  xlab("Characteristic") +
  ylab("Importance") +
  ggtitle("Characteristic importance in Random Forest")