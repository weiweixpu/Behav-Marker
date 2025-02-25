library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Set the working directory (please modify it according to your actual path)
setwd("E:\\Parkinson's Project\\Clinical\\Clinical Data")

# File path
input_files <- c("Non-Injection.csv", "First.csv", "Second.csv", "Third.csv")

# Fix feature names to match actual column names in CSV file
features <- c("Distance", "Mean speed", "Line crossings", "Center entries",
              "CENTER time", "CENTER distance", "AROUND entries", "AROUND time", "AROUND distance")

# Read CSV file and merge data
data_list <- lapply(input_files, function(file) {
  data <- read_csv(file)
  type <- if(file == "Non-Injection.csv") "Control" else "PD"
  data$Group <- type
  data %>% select(all_of(c(features, "Group")))
})

# Merge all data
combined_data <- bind_rows(data_list)

# Convert the data to long format
long_data <- combined_data %>%
  pivot_longer(cols = all_of(features), names_to = "Feature", values_to = "Value")

# Set the order of features and more readable labels
feature_order <- features
feature_labels <- c("Distance", "Mean speed", "Line crossings", "Center entries",
                    "Center time", "Center distance", "Around entries", "Around time", "Around distance")

# Convert Feature to factor and set the level order
long_data$Feature <- factor(long_data$Feature, levels = feature_order, labels = feature_labels)

# Create a graph
p <- ggplot(long_data, aes(x = Feature, y = Value, fill = Group)) +
  geom_boxplot(width = 0.7, position = position_dodge(width = 0.8)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8), 
             size = 0.5, alpha = 0.3) +
  scale_fill_manual(values = c("Control" = "#66c2a5", "PD" = "#fc8d62")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.title.x = element_blank(),
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    panel.border = element_rect(fill = NA, color = "black", size = 0.5)
  ) +
  labs(y = "Value", fill = "Group") +
  coord_flip()  # Create a graph

# Display the graph
print(p)

# Save the graph
ggsave("feature_comparison.pdf", p, width = 10, height = 8, units = "in", dpi = 300)