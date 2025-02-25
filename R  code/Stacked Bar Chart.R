# install.packages("ggplot2")
library(ggplot2)

# Create a data frame
data <- data.frame(
  Category = rep(c("Training Data", "Test Data", "Validation Data"), each = 4),
  Injection = rep(c("Control", "Third Injection", "Sixth Injection", "Tenth Injection"), times = 3),
  Count = c(755, 140, 140, 170, 364, 67, 67, 80, 88, 16, 15, 18)
)

# Set category order
data$Category <- factor(data$Category, levels = c("Training Data", "Test Data", "Validation Data"))

# Custom Colors
custom_colors <- c("Control" = "#ef7f51", "Third Injection" = "#78d3ac", 
                   "Sixth Injection" = "#9355b0", "Tenth Injection" = "#74c1f0")

# Draw a stacked bar chart
plot <- ggplot(data, aes(fill = Injection, y = Count, x = Category)) + 
  geom_bar(position = "stack", stat = "identity") +
  scale_fill_manual(values = custom_colors) +
  labs(title = "Stratified sampling", 
       y = "Number", 
       x = "Dataset Category", 
       fill = "Injection Type") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_blank()
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)))

# Show Chart
print(plot)