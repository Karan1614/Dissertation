# Load required libraries
library(glmnet)
library(caret)
library(ggplot2)
library(reshape2)

# Set the seed for reproducibility
set.seed(123)

# Define the number of observations and relevant features
num_features_total <- 20
num_examples_total <- 100

num_trials <- 100
prop_relevant_features <- 0.5
prop_train_examples <- 0.7
ci_level <- 0.95

# Define a range of regularization parameters
lambda_seq <- 10^seq(-5, 5, by = 0.5)

# Create empty matrices to store the results
train_errors <- matrix(0, nrow = num_trials, ncol = length(lambda_seq))
test_errors <- matrix(0, nrow = num_trials, ncol = length(lambda_seq))


num_relevant_features <- prop_relevant_features * num_features_total
num_train_examples <- prop_train_examples * num_examples_total 



# Perform the trials
for (i in 1:num_trials) {
  # Generate the feature matrix
  X <- matrix(rnorm(num_features_total * num_examples_total), ncol = num_features_total)
  
  # Generate the true coefficients
  true_coefficients <- c(rep(2, num_relevant_features), rep(0, (num_features_total - num_relevant_features)))
  
  # Generate the target variable
  y <- X %*% true_coefficients + rnorm(num_examples_total)
  
  # Convert data to a dataframe
  data <- data.frame(y, X)
  
  # Split the data into training and testing sets
  train_index <- createDataPartition(data$y, p = prop_train_examples, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Create an empty matrix to store the coefficients
  coefficients <- matrix(0, nrow = length(lambda_seq), ncol = dim(train_data[, -1])[2])  
  
  # Lists to store the errors for this trial
  trial_train_errors <- vector("numeric", length(lambda_seq))
  trial_test_errors <- vector("numeric", length(lambda_seq))
  bias_squared <- vector("numeric", length(lambda_seq))
  variance <- vector("numeric", length(lambda_seq))
  
  # Perform Lasso regression for each lambda value
  for (j in seq_along(lambda_seq)) {
    # Train the Lasso model
    lasso_model <- glmnet(x = as.matrix(train_data[, -1]), y = train_data$y, alpha = 1, lambda = lambda_seq[j])
    
    # Make predictions on the training set
    train_predictions <- predict(lasso_model, newx = as.matrix(train_data[, -1]))
    
    # Compute the training error
    trial_train_errors[j] <- mean((train_data$y - train_predictions)^2)
    
    # Make predictions on the test set
    test_predictions <- predict(lasso_model, newx = as.matrix(test_data[, -1]))
    
    # Compute the test error
    trial_test_errors[j] <- mean((test_data$y - test_predictions)^2)
    
    # Compute the bias squared
    bias_squared[j] <- mean((train_predictions - train_data$y)^2)
    
    # Compute the variance
    variance[j] <- mean((train_predictions - mean(train_predictions))^2)
    
    # Store the coefficients for this lambda value
    coefficients[j, ] <- coef(lasso_model)[-1]
    
  }
  
  
  # Store the errors for this trial in the matrices
  train_errors[i, ] <- trial_train_errors
  test_errors[i, ] <- trial_test_errors
  
}

# Calculate the mean and standard deviation of the train and test errors
mean_train_errors <- apply(train_errors, 2, mean)
mean_test_errors <- apply(test_errors, 2, mean)
sd_train_errors <- apply(train_errors, 2, sd)
sd_test_errors <- apply(test_errors, 2, sd)

# Create a data frame with lambda and errors
error_data <- data.frame(lambda = lambda_seq, TrainError = mean_train_errors, TestError = mean_test_errors, TrainSE = sd_train_errors, TestSE = sd_test_errors)

# Plot the train and test errors with confidence bands
ggplot(error_data,aes(x = log10(lambda))) +
  geom_line(aes(y = TrainError, color = "Train Error")) +
  geom_ribbon(aes(ymin = TrainError - qnorm(0.5*(1 + ci_level)) * TrainSE/sqrt(num_trials - 1), ymax = TrainError + qnorm(0.5*(1 + ci_level)) * TrainSE/sqrt(num_trials - 1), fill = "Train Error"), alpha = 0.2) +
  geom_line(aes(y = TestError, color = "Test Error")) +
  geom_ribbon(aes(ymin = TestError - qnorm(0.5*(1 + ci_level)) * TestSE/sqrt(num_trials - 1), ymax = TestError + qnorm(0.5*(1 + ci_level)) * TestSE/sqrt(num_trials - 1), fill = "Test Error"), alpha = 0.2) +
  scale_color_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  scale_fill_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  labs(x = "log10(lambda)", y = "Error", title = "Mean Train and Test Errors with Confidence Bands") +
  theme_minimal() 

# Create a data frame with lambda and bias/variance
bias_variance_data <- data.frame(lambda = lambda_seq, BiasSquared = bias_squared, Variance = variance)

# Plot the bias squared and variance
ggplot(bias_variance_data, aes(x = log10(lambda))) +
  geom_line(aes(y = BiasSquared, color = "Bias Squared")) +
  geom_line(aes(y = Variance, color = "Variance")) +
  scale_color_manual(values = c("Bias Squared" = "blue", "Variance" = "red")) +
  labs(x = "log10(lambda)", y = "Bias Squared / Variance") +
  theme_minimal()


# Create a data frame with lambda and coefficients
coefficients_data <- data.frame(lambda = lambda_seq, coefficients)

# Melt the data frame to long format for plotting
melted_coefficients <- reshape2::melt(coefficients_data, id.vars = "lambda")

# Plot the coefficients as a function of the hyperparameter
ggplot(melted_coefficients, aes(x = log10(lambda), y = value, color = variable)) +
  geom_line() +
  labs(x = "log10(lambda)", y = "Coefficient Value", title = "Coefficients as a Function of Hyperparameter") +
  theme_minimal()
















