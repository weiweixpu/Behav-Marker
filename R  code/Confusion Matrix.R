library(caret)
library(ggplot2)
library(gridExtra)
library(grid)
library(scales)

# Read CSV file
file_path <- "E:/Parkinson's Project/data/results/plus 1007 data/complete results/basic model/big_model_TM_test_predictions.csv"
data <- read.csv(file_path)

# Assume there are two columns in the dataset: True.Label and Class_1_Probability
true_labels <- ifelse(data$True.Label == 1, 1, 0)
predicted_probabilities <- data$Class_1_Probability

# Generate predicted labels based on threshold 0.5
predicted_labels <- ifelse(predicted_probabilities >= 0.5, 1, 0)

# Create confusion matrix
confusion_mat <- table(Predicted = predicted_labels, Actual = true_labels)

# Calculate sensitivity and specificity
sensitivity <- confusion_mat[2, 2] / sum(confusion_mat[2, ])
specificity <- confusion_mat[1, 1] / sum(confusion_mat[1, ])

# Prepare confusion matrix data
cm_table <- as.data.frame(confusion_mat)
cm_table$Percent <- round((cm_table$Freq / sum(cm_table$Freq)) * 100, 1)

# Add labels
cm_table$Actual_Label <- ifelse(cm_table$Actual == 0, "Control", "PD")
cm_table$Predicted_Label <- ifelse(cm_table$Predicted == 0, "Control", "PD")

# Create a custom label function
label_function <- function(x) {
sprintf("%d\n(%.1f%%)", x$Freq, x$Percent)
}

# Plot the confusion matrix
cm_plot <- ggplot(cm_table, aes(Actual_Label, Predicted_Label)) +
  geom_tile(aes(fill = Freq), color = "white", size = 1.5) +
  geom_text(aes(label = label_function(cm_table)), size = 5, fontface = "bold") +
  scale_fill_gradient(low = "#FFF5F0", high = "#67000D") +
  labs(
       title = "Confusion matrix of foundation trajectory graph model",
       # title = "Confusion matrix of randomly initialized Densenet121 heatmap model"pretrained trajectory graph model,
       x = "Actual",
       y = "Predicted") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, size = 1.5)
  ) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0))


final_plot <- grid.arrange(cm_plot, nrow = 2, heights = c(4, 0.5))

# Save the chart
# ggsave("confusion_matrix_labeled.png", final_plot, width = 10, height = 8, dpi = 300)