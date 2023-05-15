library(ggplot2)

# Read data
data("iris")

# Explore data
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + 
 geom_point() + xlab("Petal length") + ylab("Petal width")
summary(iris$Species)

# Train model
k <- 2
cluster <- kmeans(iris[, 3:4], k)
iris$cluster <- cluster$cluster
iris$cluster <- as.factor(iris$cluster)

# Explore model
ggplot(iris, aes(Petal.Length, Petal.Width, color = cluster)) +
 geom_point() + xlab("Petal length") + ylab("Petal width")

# Setosa correctly isolated. Virginica and versicolor mixed
# Create a table and elbow function to explore the results from k 1 to 10
clusterTable <- table(iris$cluster, iris$Species)
elbowfun <- function(df) {
  Num_k <- rep(NA, 10)
  Error <- rep(NA ,10)
  for (i in 1:10) {
    cluster <- kmeans(df, i)
    Num_k[i] <- i
    Error[i] <- cluster$tot.withinss
  }
  df_new <- data.frame(Num_k, Error)
  return (df_new)
}
df = iris[, 3:4]
Elbow <- elbowfun(df) 
# Draw elbow function chart
ggplot(Elbow, aes(Num_k, Error)) + geom_line() +
  geom_point(color = "blue", size = 3) + xlab("Num_k") + ylab ("Error")
# Conclusion: drastical error reduction using 2, and still reducing with 3.
# No variation higher than 3 due to minimal variance

# Apply k = 3
k <- 3
cluster <- kmeans(iris[, 3:4], k)
iris$cluster <- cluster$cluster
iris$cluster <- as.factor(iris$cluster)

# Draw it
ggplot(iris, aes(Petal.Length, Petal.Width, color = cluster)) +
  geom_point() + xlab("Petal length") + ylab("Petal width")