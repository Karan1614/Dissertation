# Load necessary libraries
library(readxl)
library(knockoff)
library(dplyr)
library(ggplot2)

# Load the data
setwd("C:/Users/Karan/Desktop/Dissertation")
df <- read_excel("Player_Valuation.xlsx")

# Drop non-numeric columns
df <- df[, sapply(df, is.numeric)]

# Handle missing values by imputation or removal
df <- na.omit(df)

# Split into features (X) and target variable (y)
y <- df$Market_Value
X <- df %>% select(-Ranking, -Market_Value)

# Initialize an empty data frame to store results
results_df <- data.frame()

# Loop over different FDR values
fdr_values <- seq(0, 1, by = 0.05)

for (fdr in fdr_values) {
  selected_vars <- knockoff.filter(X, y, fdr = fdr)
  
  # Create a temporary data frame to store the results
  temp_df <- data.frame(FDR = fdr, SelectedVars = length(selected_vars$selected))
  
  # Append to the results data frame
  results_df <- rbind(results_df, temp_df)
}

# Plot the number of selected variables against FDR values
ggplot(results_df, aes(x = FDR, y = SelectedVars)) +
  geom_line() +
  geom_point() +
  xlab("FDR Value") +
  ylab("Number of Selected Variables") +
  ggtitle("Number of Selected Variables for Different FDR Values") +
  theme_minimal()
View(selected_vars)
