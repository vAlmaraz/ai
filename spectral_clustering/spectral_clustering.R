library(mlbench)
library(kernlab)

# Create data
initial_data <- mlbench.spirals(100, 1, 0.025)
data <- 4 * initial_data$x

# Explore data
plot(data)

# Train the model
model <- specc(data, centers = 2)

# Explore the model
plot(data, col = model, pch = 4)

# Fix results in order to compare later
points(data, col = initial_data$classes, pch = 5)

# Compare using k-means
cluster_kmeans <- kmeans(data, 2)

# Draw it
plot(data, col = cluster_kmeans$cluster, pch = 4)
# It groups by physical position rather than tendency

# Draw initial points to compare
points(data, col = initial_data$classes, pch = 5)