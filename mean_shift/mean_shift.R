library(ggplot2)
library(meanShiftR)

# Read data
data("iris")

# Explore data
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) +
  geom_point() + xlab("Petal length") + ylab ("Petal width")

petals <- cbind(iris$Petal.Length, iris$Petal.Width)

# Train the model
tags <- meanShift(queryData = petals, trainData = petals, 
                  bandwidth = c(0.7, 0.7))
iris$tags <- as.factor(tags$assignment)

# Explore the model
ggplot(iris, aes(Petal.Length, Petal.Width, color = tags)) +
  geom_point() + xlab("Petal length") + ylab ("Petal width")

MC <- table(iris$tags, iris$Species)

# Get model accuracy
sum(diag(MC))/sum(MC)

# Draw it
centroids <- unique(tags$value)
ggplot(iris, aes(Petal.Length, Petal.Width, color = tags)) +
  geom_point() + xlab("Petal length") + ylab ("Petal width") +
  geom_point(aes(x= centroids[1,1], y= centroids[1,2]),
             colour="red",shape = 24)+
  geom_point(aes(x= centroids[2,1], y= centroids[2,2]),
             colour="green",shape = 24)+
  geom_point(aes(x= centroids[3,1], y= centroids[3,2]),
             colour="blue",shape = 24)
