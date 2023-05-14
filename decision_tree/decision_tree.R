library("rpart")
library("rpart.plot")

# Read data
iris <- datasets::iris

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_index <- sample(nrow(iris), size = round(0.7*nrow(iris)), replace = FALSE)
train <- iris[train_index, ]
test <- iris[-train_index, ]

# Train the model
model <- rpart(Species ~ ., data = train)

# Test the model
predictions <- predict(model, test, type = "class")

# Get model accuracy
accuracy <- sum(predictions == test$Species) / nrow(test)
print(paste0("Model accuracy on test data: ", round(accuracy, 2)))

# Draw it
print(model)
plot(model, uniform = TRUE, main = "Decision tree for Iris data")
text(model, use.n = TRUE, all = TRUE, cex = 0.8)