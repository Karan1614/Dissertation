library(readxl)
library(tidyverse)

setwd("C:/Users/Karan/Desktop/Dissertation/")
data <- read_excel("Final.xlsx")

# Check for missing values in the dataset
missing_values = data %>%
  is.na() %>%
  colSums()

# Print the number of missing values for each variable
print(missing_values)

