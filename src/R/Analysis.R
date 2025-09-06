install.packages("tidyverse")

library(tidyverse)
library(ggplot2)
library(readr)

df<- read_csv("/home/comp/Downloads/german_credit.csv")
glimpse(data)
class(df)
counts <- table(df$Creditability)

summary(df)

#Histograms

ggplot(df, aes(x = `Duration of Credit (month)`)) +
  geom_histogram(aes(y = ..density..), binwidth = 3, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1.2) +
  labs(title = "Histogram with Density Curve",
       x = "Duration of Credit (Months)",
       y = "Density") +
  theme_minimal()

#Boxplots

ggplot(df, aes(x = factor(Creditability), y = `Duration of Credit (month)`)) +
  geom_boxplot(fill = "lightgreen", color = "black") +
  labs(title = "Boxplot of Duration of Credit (month) by Creditability",
       x = "Creditability (0 = Not Creditworthy, 1 = Creditworthy)",
       y = "Duration of Credit (month)") +
  theme_minimal()
#pie charts
library(ggplot2)

# Create data frame
pie_data <- data.frame(
  category = names(counts),
  count = as.numeric(counts),
  percentage = round(prop.table(counts)*100, 1)
)
print(pie_data)
