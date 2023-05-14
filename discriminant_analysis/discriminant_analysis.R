# Read data
data(iris)
irisData <- iris

X <- iris[, 1:4]
y <- iris[, 5]

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
train_index <- sample(1:nrow(X), size = nrow(X)*0.7, replace = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Train the model
require(MASS)
model <- lda(Species ~ ., data = data.frame(X_train, Species = y_train))

# Test the model
y_pred <- predict(model, newdata = X_test)

# Get model accuracy
mean(y_pred$class == y_test)

# Draw it
par(mfrow = c(2, 2))

original <- cbind(X_test, y_test)
colnames(original) <- c(colnames(X_test), "Species")
# New model just for drawing the plot
model1 <- lda(Species ~ ., data = data.frame(X_test, Species = y_test))
color <- rep("green",nrow(original))
color[original$Species == "setosa"] <- "red"
color[original$Species == "virginica"] <- "blue"
plot(model1, dimen=2, col=color, abbrev=3)

prediction <- cbind(X_test, y_pred$class)
colnames(prediction) <- c(colnames(X_test), "Species")
newY <- prediction[, 5]
# New model just for drawing the plot
model2 <- lda(Species ~ ., data = data.frame(X_test, Species = newY))
color <- rep("green",nrow(prediction))
color[prediction$Species == "setosa"] <- "red"
color[prediction$Species == "virginica"] <- "blue"
plot(model2, dimen=2, col=color, abbrev=3)
