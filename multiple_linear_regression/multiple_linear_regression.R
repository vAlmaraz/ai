# https://rstudio-pubs-static.s3.amazonaws.com/876485_f4e5224c074c46999d47ac2e7ca1cf0e.html
library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)

# Read data
heartData <- read.csv("../datasets/heart.data.csv")
# Check data
summary(heartData)
# Check for independence of observations (aka no autocorrelation)
# The correlation of 1.5% is small enough that we can include both variables in the model
cor(heartData$biking, heartData$smoking)
# Check for normality of dependent variable (heart disease) with a histogram
# The distribution of observations is roughly bell shaped so we can proceed
hist(heartData$heart.disease)
# Check for linearity between heart disease and biking, and disease and smoking
# Both look roughly linear
plot(heart.disease ~ biking, data = heartData)
plot(heart.disease ~ smoking, data = heartData)

# Train the model
model <- lm(heart.disease ~ biking + smoking, data = heartData)
summary(model)
# Check for homoscedasticity
# The red lines representing the mean of the residuals are all basically horizontal and centered around zero.
# This means there are no outliers or biases in the data that would make a linear regression invalid.
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Draw it
plotting.data <- expand.grid(
    biking = seq(min(heartData$biking), max(heartData$biking), length.out = 30),
    smoking = c(
        min(heartData$smoking), mean(heartData$smoking),
        max(heartData$smoking)
    )
)
plotting.data$predicted.y <- predict.lm(model, newdata = plotting.data)
plotting.data$smoking <- round(plotting.data$smoking, digits = 2)
plotting.data$smoking <- as.factor(plotting.data$smoking)
heart.plot <- ggplot(heartData, aes(x = biking, y = heart.disease)) +
    geom_point()
heart.plot <- heart.plot +
    geom_line(data = plotting.data, aes(x = biking, y = predicted.y, color = smoking), size = 1.25)
heart.plot
heart.plot +
  theme_bw() +
  labs(title = "Rates of heart disease (% of population) \n as a function of biking to work and smoking",
       x = "Biking to work (% of population)",
       y = "Heart disease (% of population)",
       color = "Smoking \n (% of population)")

heart.plot
