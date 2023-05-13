# Read Weather data file
setwd(getwd())
pathFile <- file.path("weather.csv")
weatherData <- read.csv(pathFile, stringsAsFactors = FALSE)

# Ignore dates (in order to prevent error when predicting)
# Error: factor Date has new levels
weatherData <- subset(weatherData, select = -c(Date))

# Omit rows without available values
weatherData <- na.omit(weatherData)

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 70% of data
randomIndexes <- sample(1:nrow(weatherData))
splitRate <- floor(0.7 * nrow(weatherData))
trainRate <- randomIndexes[1:splitRate]
# Split data into train and test
trainData <- weatherData[trainRate, ]
testData <- weatherData[-trainRate, ]

# Convert RainTomorrow into 0 and 1 instead of No and Yes
trainData$RainTomorrow <- ifelse(trainData$RainTomorrow == "Yes", 1, 0)


model <- glm( RainTomorrow ~., data = trainData, family = binomial)
# POSSIBLE ERROR: 
# Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]) : contrasts can be applied only to factors with 2 or more levels
# Why? There is a column with a unique value for all rows:
# https://www.statology.org/contrasts-applied-to-factors-with-2-or-more-levels/
# Investigate different values per col:
# sapply(lapply(trainData, unique), length)

# Test the model
probabilities <- predict(model, testData, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "Yes", "No")

# Sacamos la precisión del modelo de test
mean(predicted.classes == testData$RainTomorrow)