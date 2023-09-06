# Load necessary libraries
library(readxl)
library(dplyr)
library(glmnet)

# Load the dataset
data <- read_excel("Player_Valuation.xlsx")

# Remove rows with missing values
data_clean <- na.omit(data)

# Separate predictors and response
X <- data_clean %>% select(-Ranking, -Player, -Market_Value)
y <- data_clean$Market_Value

# Encode categorical variables
X_encoded <- model.matrix(~., data = X)

# Perform cross-validation for Ridge Regression
set.seed(123) # For reproducibility
cv_fit <- cv.glmnet(X_encoded, y, alpha=0)  # alpha=0 for Ridge Regression

# Extract the optimal lambda value
optimal_lambda <- cv_fit$lambda.min

# Print the optimal lambda
optimal_lambda

# Compute Ridge Regression coefficients using the formula for the optimal lambda
beta_ridge <- solve(t(X_encoded) %*% X_encoded + optimal_lambda * diag(ncol(X_encoded))) %*% t(X_encoded) %*% y

beta_df <- as.data.frame(beta_ridge)
beta_df$Variable <- rownames(beta_df)
colnames(beta_df)[1] <- "Coefficient"

# Plot the coefficients
ggplot(beta_df, aes(x=Variable, y=Coefficient)) +
  geom_bar(stat='identity') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Ridge Regression Coefficients",
       x = "Variables",
       y = "Coefficient Value")

