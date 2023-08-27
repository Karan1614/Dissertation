# Load required libraries
library(glmnet)
library(caret)
library(reshape2)
library(ggplot2)

setwd("C:/Users/Karan/Desktop/Dissertation/")
data <- read_excel("Player_Valuation.xlsx")
# Remove rows with missing data using complete.cases()
cleaned_data <- data[complete.cases(data), ]

# Define the number of trials
num_trials <- 100
ci_level <- 0.95


# Initialize matrices to store errors for each trial
train_errors <- matrix(0, nrow = num_trials, ncol = length(lambda_seq))
test_errors <- matrix(0, nrow = num_trials, ncol = length(lambda_seq))

# Perform ridge regression over 100 trials
for (trial in 1:num_trials) {
  # Split the data into training, validation, and test sets (60-20-20 split)
  set.seed(123) # Set a seed for reproducibility
  train_index <- createDataPartition(cleaned_data$Market_Value, p = 0.6, list = FALSE)
  train_data <- cleaned_data[train_index, ]
  remaining_data <- cleaned_data[-train_index, ]
  validation_index <- createDataPartition(remaining_data$Market_Value, p = 0.5, list = FALSE)
  validation_data <- remaining_data[validation_index, ]
  test_data <- remaining_data[-validation_index, ]
  
  # Identify numeric columns in the dataset
  numeric_cols <- sapply(train_data, is.numeric)
  
  # Prepare the matrices for ridge regression
  X_train <- as.matrix(train_data[, numeric_cols])
  View(X_train)
  y_train <- as.matrix(as.numeric(train_data$Market_Value))
  View(y_train)
  X_validation <- as.matrix(validation_data[, numeric_cols])
  y_validation <- validation_data$Market_Value
  X_test <- as.matrix(test_data[, numeric_cols])
  y_test <- test_data$Market_Value
  
  #Perform feature scaling (normalize the features)
  X_train_scaled <- scale(X_train, center = TRUE, scale = TRUE)
  X_validation_scaled <- scale(X_validation, center = TRUE, scale = TRUE)
  X_test_scaled <- scale(X_test, center = TRUE, scale = TRUE)
  
  # Check for NAs and non-numeric values in the target variable (y_train)
  if (anyNA(y_train) || !is.numeric(y_train)) {
    stop("Target variable (y_train) has missing or non-numeric values. Please check the data.")
  }
  
  
  # Perform ridge regression with cross-validation to find the optimal lambda value
  lambda_seq <- 10^seq(-3, 3, by = 0.1) # Define a range of lambda values
  ridge_model <- glmnet(x = X_train_scaled, y = y_train, alpha = 0, lambda = lambda_seq, nfolds = 5)

  # Find the optimal lambda value based on cross-validation results
  lambda_optimal <- ridge_model$lambda.min

  # Train the final ridge model using the optimal lambda value and the full training set
  final_ridge_model <- glmnet(x = X_train_scaled, y = y_train, alpha = 0, lambda = lambda_optimal)
  # Make predictions on the training set using the final ridge model
  train_predictions <- predict(final_ridge_model, newx = X_train_scaled)
  
  # Calculate the training error (mean squared error)
  train_errors[trial, ] <- apply(predict(ridge_model, s = lambda_seq, newx = X_train_scaled), 2, mean)
  
  # Make predictions on the validation set using the final ridge model
  validation_predictions <- predict(final_ridge_model, newx = X_validation_scaled)
  
  # Calculate the validation error (mean squared error)
  test_errors[trial, ] <- apply(predict(ridge_model, s = lambda_seq, newx = X_validation_scaled), 2, mean)

}

#Calculate the mean and standard deviation of train and test errors across trials
mean_train_errors <- apply(train_errors, 2, mean)

mean_test_errors <- apply(test_errors, 2, mean)
sd_train_errors <- apply(train_errors, 2, sd)
sd_test_errors <- apply(test_errors, 2, sd)

#Create a data frame with lambda and errors
error_data <- data.frame(lambda = lambda_seq, TrainError = mean_train_errors, TestError = mean_test_errors, TrainSE = sd_train_errors, TestSE = sd_test_errors)

#Plot the train and test errors with confidence bands
ggplot(error_data,aes(x = log10(lambda))) +
  geom_line(aes(y = TrainError, color = "Train Error")) +
  geom_ribbon(aes(ymin = TrainError - qnorm(0.5*(1 + ci_level)) * TrainSE/sqrt(num_trials - 1), ymax = TrainError + qnorm(0.5*(1 + ci_level)) * TrainSE/sqrt(num_trials - 1), fill = "Train Error"), alpha = 0.2) +
  geom_line(aes(y = TestError, color = "Test Error")) +
  geom_ribbon(aes(ymin = TestError - qnorm(0.5*(1 + ci_level)) * TestSE/sqrt(num_trials - 1), ymax = TestError + qnorm(0.5*(1 + ci_level)) * TestSE/sqrt(num_trials - 1), fill = "Test Error"), alpha = 0.2) +
  scale_color_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  scale_fill_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  labs(x = "log10(lambda)", y = "Error", title = "Mean Train and Test Errors with Confidence Bands") +
  theme_minimal() 

#Plot coefficients as a function of lambda
ridge_model <- glmnet(x = X_train_scaled, y = y_train, alpha = 0, lambda = lambda_seq)
coef_data <- as.data.frame(coef(ridge_model))
coef_data$lambda <- lambda_seq
coef_data_long <- melt(coef_data, id.vars = "lambda")
ggplot(coef_data_long, aes(x = lambda, y = value, color = variable)) + 
  geom_line() +
  labs(x = "Regularization Parameter (lambda)", y = "Coefficients") +
  theme_minimal()

#Plot bias squared and variance as a function of lambda
bias_squared <- rowSums((mean_train_errors - train_errors)^2) / num_trials
variance <- rowSums((train_errors - matrix(rep(mean_train_errors, num_trials), nrow = num_trials, byrow = FALSE))^2) / (num_trials - 1)
plot(lambda_seq, bias_squared, type = "l", col = "blue", xlab = "Regularization Parameter (lambda)", ylab = "Bias Squared")
lines(lambda_seq, variance, type = "l", col = "red")
legend("topright", legend = c("Bias Squared", "Variance"), col = c("blue", "red"), lty = 1)
