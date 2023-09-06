# Load necessary libraries
library(readxl)
library(knockoff)
library(dplyr)

# Load the data
setwd("C:/Users/Karan/Desktop/Dissertation")
df <- read_excel("Player_Valuation.xlsx")
View(df)
# 2. Preprocess the Data
# Drop non-numeric columns
df <- df[, sapply(df, is.numeric)]

# Handle missing values by imputation or removal
df <- na.omit(df)

# Split into features (X) and target variable (y)
y <- df$Market_Value
X <- df %>% select(-Ranking, -Market_Value)

# 4. Fit a Model and Select Variables
# Using knockoff.filter for both fitting the model and variable selection
selected_vars <- knockoff.filter(X, y, fdr = 0.20)
print(selected_vars)
View(selected_vars)

