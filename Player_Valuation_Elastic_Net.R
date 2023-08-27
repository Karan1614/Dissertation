# Load necessary libraries
library(glmnet)
library(caret)

# Prepare the data (as before)
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
y_train <- y_train_val[trainIndex]
X_val <- X_train_val[-trainIndex,]
y_val <- y_train_val[-trainIndex] 
# Define a grid of hyperparameters to test
alpha_grid <- seq(0, 1, by=0.1) # Varying alpha from 0 (Ridge) to 1 (Lasso)
lambda_grid <- 10^seq(-6, 6, by=1) # Range of lambda values

# Store the results
results <- data.frame()

# Loop through alpha values
for (alpha_value in alpha_grid) {
  # Perform cross-validation for elastic net
  cv_model <- cv.glmnet(X_train, y_train, alpha = alpha_value, lambda = lambda_grid)
  
  # Collect results
  results <- rbind(results, data.frame(Alpha=alpha_value, Lambda=cv_model$lambda.min, MSE=cv_model$cvm[cv_model$lambda == cv_model$lambda.min]))
}

# Find the optimal alpha and lambda
optimal_result <- results[which.min(results$MSE),]
optimal_alpha <- optimal_result$Alpha
optimal_lambda <- optimal_result$Lambda

# Print the optimal alpha and lambda
print(paste("Optimal alpha:", optimal_alpha))
print(paste("Optimal lambda:", optimal_lambda))

# Range of lambda values
lambdas <- 10^seq(-6, 6, by = 1)

# Data frame to store results
results <- data.frame()

# Loop through lambda values
for (alpha in lambdas) {
  model <- glmnet(X_train, y_train, alpha = 0.3, lambda = alpha)
  
  # Train and Test Errors
  train_pred <- predict(model, X_train)
  test_pred <- predict(model, X_test)
  train_error <- mean((y_train - train_pred)^2)
  test_error <- mean((y_test - test_pred)^2)
  
  # Bias Squared and Variance
  predictions <- predict(model, X_val)
  bias_sq <- mean((y_val - predictions)^2)
  variance <- mean((predictions - mean(predictions))^2)
  
  # Add to results data frame
  results <- rbind(results, data.frame(Lambda=alpha, TrainError=train_error, TestError=test_error, BiasSquared=bias_sq, Variance=variance))
}

# Plot Train and Test Errors vs Lambda (with confidence bands)
ggplot(results, aes(x=log10(Lambda))) +
  geom_line(aes(y=TrainError, color="Train Error")) +
  geom_line(aes(y=TestError, color="Test Error")) +
  geom_ribbon(aes(ymin=TrainError-1.96*sqrt(TrainError), ymax=TrainError+1.96*sqrt(TrainError), fill="Train Error"), alpha=0.2) +
  geom_ribbon(aes(ymin=TestError-1.96*sqrt(TestError), ymax=TestError+1.96*sqrt(TestError), fill="Test Error"), alpha=0.2) +
  scale_color_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  scale_fill_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  labs(x = "log10(lambda)", y = "Error", title = "Mean Train and Test Errors with Confidence Bands") +
  theme_minimal() 

# Plot Bias Squared and Variance vs Lambda
ggplot(results, aes(x=log10(Lambda))) +
  geom_line(aes(y=BiasSquared, color="Bias Squared")) +
  geom_line(aes(y=Variance, color="Variance")) +
  labs(title="Bias Squared and Variance vs Lambda", x="Lambda", y="Value") +
  scale_color_manual(values=c("Bias Squared"="blue", "Variance"="red"))


# Initialize a matrix to store coefficients
coefficients <- matrix(0, nrow=length(lambdas), ncol=ncol(X_encoded))

# Loop through lambda values to collect coefficients
for (i in seq_along(lambdas)) {
  alpha <- lambdas[i]
  model <- glmnet(X_train, y_train, alpha = 0, lambda = alpha)
  coefficients[i,] <- as.numeric(coef(model)[-1]) # Removing intercept
}

# Plot Coefficients vs Lambda using matplot
matplot(lambdas, coefficients, type="l", log="x", xlab="Lambda", ylab="Coefficient Value", main="Coefficients vs Lambda")
