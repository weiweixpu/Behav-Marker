library(ggpubr)
library(dplyr)

# File path and output file name
input_files <- c("Non-Injection.csv", "Third.csv", "Sixth.csv", "Tenth.csv")
setwd("E:/Parkinson's Project/Paper/Figure 2")

# Specify the column name to read
distance_column <- "AROUND distance"

# Read CSV file and merge data
data_list <- lapply(input_files, function(file) {
  data <- read.csv(file, check.names=FALSE)
  type <- sub("\\.csv", "", basename(file))
  data$Type <- type
  return(data[, c(distance_column, "Type")])
})

# Merge all data
combined_data <- do.call(rbind, data_list)
colnames(combined_data)[1] <- "AROUND distance"

# Make sure "Non-Injection" is first
group <- c("Non-Injection", "Third", "Sixth", "Tenth")
combined_data$Type <- factor(combined_data$Type, levels=group)

# Set up comparison groups
comp <- combn(group, 2)
my_comparisons <- list()
for (i in 1:ncol(comp)) {
  my_comparisons[[i]] <- comp[, i]
}

# Draw a violin plot and display it directly
p <- ggviolin(combined_data, x="Type", y="AROUND distance", fill = "Type", 
              xlab="Injection", ylab="Around distance (m)", 
              legend.title="Injection type", 
              add = "boxplot", add.params = list(fill="white")) + 
  stat_compare_means(comparisons = my_comparisons)

# Display the image
print(p)