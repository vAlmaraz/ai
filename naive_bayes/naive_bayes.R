library(dplyr)
library(tidyr)
library(caret)
library(e1071)

# Read Weather data file
weatherData <- read.csv('../datasets/weather.csv')

# Ignore dates (in order to prevent error when predicting)
weatherData <- select(weatherData, -Date)

# Omit rows without available values
weatherData <- na.omit(weatherData)

# Convert RainToday and RainTomorrow into 0 and 1 instead of No and Yes
weatherData$RainToday <- ifelse(weatherData$RainToday == "Yes", 1, 0)
weatherData$RainTomorrow <- ifelse(weatherData$RainTomorrow == "Yes", 1, 0)

X <- select(weatherData, -RainTomorrow)
y <- weatherData$RainTomorrow

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
indexes <- sample(1:nrow(weatherData))
train_size <- floor(nrow(weatherData) * 0.7)
train_idx <- indexes[1:train_size]
test_idx <- indexes[(train_size + 1):nrow(weatherData)]
# Split data into train and test
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[test_idx, ]
y_test <- y[test_idx]

# Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might end up creating a bias
sc_x <- preProcess(X_train, method=c("center", "scale"))
X_train <- predict(sc_x, X_train)
X_test <- predict(sc_x, X_test)

# Train the model
model <- naiveBayes(X_train, y_train)

# Test the model
y_pred <- predict(model, X_test)

# Get model accuracy
accuracy <- sum(y_test == y_pred) / length(y_test)
print(paste("Model accuracy on test data:", round(accuracy, 4)))

# Draw it
# TODO