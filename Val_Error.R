# Load necessary libraries
library(readxl)
library(dplyr)
library(tidyr)
library(caret)
library(glmnet)
library(ggplot2)
library(reshape2)

# Load the data
setwd("C:/Users/Karan/Desktop/Dissertation")
data <- read_excel("Player_Valuation.xlsx")

# Remove rows with missing values
data_clean <- na.omit(data)

# Separate predictors and response
X <- data_clean %>% select(-Ranking, -Player, -Market_Value)
y <- data_clean$Market_Value

# Encode categorical variables
X_encoded <- model.matrix(~., data = X)

prop_train_examples <- 0.6

# Split the data
set.seed(42)
trainIndex <- createDataPartition(y, p = prop_train_examples, list = FALSE)
X_train_val <- X_encoded[trainIndex,]
y_train_val <- y[trainIndex]
X_test <- X_encoded[-trainIndex,]
y_test <- y[-trainIndex]

trainIndex <- createDataPartition(y_train_val, p = 0.75, list = FALSE)
X_train <- X_train_val[trainIndex,]
y_train <- as.matrix(y_train_val[trainIndex])
X_val <- X_train_val[-trainIndex,]
y_val <- y_train_val[-trainIndex]

# Range of lambda values
lambdas <- 10^seq(-6, 6, by = 1)



# Data frame to store results
results <- data.frame()

# Loop through lambda values
for (alpha in lambdas) {
  model <- glmnet(X_train, y_train, alpha = 0.3, lambda = alpha)
  
  # Train, Validation, and Test Errors
  train_pred <- predict(model, X_train)
  val_pred <- predict(model, X_val)
  test_pred <- predict(model, X_test)
  
  train_error <- mean((y_train - train_pred)^2)
  val_error <- mean((y_val - val_pred)^2)
  test_error <- mean((y_test - test_pred)^2)
  
  # Bias Squared and Variance
  bias_sq <- mean((y_train - train_pred)^2)
  variance <- mean((train_pred - mean(train_pred))^2)
  
  # Add to results data frame
  results <- rbind(results, data.frame(Lambda=alpha, TrainError=train_error, ValError=val_error, TestError=test_error, BiasSquared=bias_sq, Variance=variance))
}

# Plot Train, Validation, and Test Errors vs Lambda (with confidence bands)
ggplot(results, aes(x=log10(Lambda))) +
  geom_line(aes(y=TrainError, color="Train Error")) +
  geom_line(aes(y=ValError, color="Validation Error")) +
  geom_line(aes(y=TestError, color="Test Error")) +
  geom_ribbon(aes(ymin=TrainError-1.96*sqrt(TrainError), ymax=TrainError+1.96*sqrt(TrainError), fill="Train Error"), alpha=0.2) +
  geom_ribbon(aes(ymin=ValError-1.96*sqrt(ValError), ymax=ValError+1.96*sqrt(ValError), fill="Validation Error"), alpha=0.2) +
  geom_ribbon(aes(ymin=TestError-1.96*sqrt(TestError), ymax=TestError+1.96*sqrt(TestError), fill="Test Error"), alpha=0.2) +
  scale_color_manual(values = c("Train Error" = "blue", "Validation Error" = "green", "Test Error" = "red")) +
  scale_fill_manual(values = c("Train Error" = "blue", "Validation Error" = "green", "Test Error" = "red")) +
  labs(x = "log10(lambda)", y = "Error", title = "Mean Train, Validation, and Test Errors with Confidence Bands") +
  theme_minimal()
