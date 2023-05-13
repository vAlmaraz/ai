library("e1071")

# Read Cats data file
setwd(getwd())
pathFile <- file.path("dataset-32337.csv")
catsData <- read.csv(pathFile, stringsAsFactors = FALSE)

# Convert Sex into 0 and 1 instead of M and F
catsData$Sex <- ifelse(catsData$Sex == "F", 1, 0)

set.seed(123)
# Calculate random indexes and split into two groups. Train data will include 60% of data
indexes <- sample(2, nrow(catsData), replace=TRUE, prob=c(0.6, 0.4))
# Split data into train and test
trainData <- catsData[indexes==1,]
testData <- catsData[indexes==2,]

# Train the model
model <- svm(Sex~., data=trainData, kernel="radial", type="C")
# Possible error:
# Error in if (any(as.integer(y) != y)) stop("dependent variable has to be of factor or integer type for classification mode.") : 
# missing value where TRUE/FALSE needed
# Why? Sex is M or F instead of factor or integer
# Solution: 
# Convert Sex into 0 or 1 for example
# https://community.rstudio.com/t/problems-when-creating-svm-model-with-linear-kernel/64975

# Kernel:
# linear, polynomial, radial, sigmoid
# Type:
# C-classification, nu-classification, one-classification (novelty detection), eps-regression, nu-regression

# Test the model
predicted <- predict(model, newdata=testData[-1])

# Get model accuracy
accuracy1 <- mean(predicted == testData$Sex)
# Or
MC <- table(testData[,1], predicted)
accuracy2 <- (sum(diag(MC)))/sum(MC)

# Draw it
plot(model, catsData)


