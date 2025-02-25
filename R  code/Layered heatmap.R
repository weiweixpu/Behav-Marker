library(corrplot)
library(pheatmap)

# Set the working directory
setwd("E:\\Parkinson's Project\\Clinical")

# Read CSV file
inputFile <- "Clinical data.csv"

# Set character encoding options
options(encoding = "UTF-8")
Sys.setlocale("LC_ALL", "Chinese")

# Read the file content to check the encoding
file_content <- readLines(inputFile, encoding = "UTF-8")
cat("First few lines of the file:\n", head(file_content), "\n")

# Try to read CSV file with UTF-8 encoding
rt <- try(read.csv(inputFile, header = TRUE, row.names = 1, fileEncoding = "UTF-8"), silent = TRUE)
if ("try-error" %in% class(rt)) {
# If UTF-8 reading fails, try GBK encoding
rt <- try(read.csv(inputFile, header = TRUE, row.names = 1, fileEncoding = "GBK"), silent = TRUE)
}

# Check if the file was read successfully
if ("try-error" %in% class(rt)) {
stop("Unable to read the file, please check the file encoding or content.")
} else {
cat("File read successfully.\n")
}

# Specify the columns to select
selected_columns <- c("Distance", "Mean.speed", "Line.crossings", "Center.entries", 
                      "CENTER.time", "CENTER.distance", "AROUND.entries", 
                      "AROUND.time", "AROUND.distance","label")

# Check if the selected column exists in the dataset
selected_columns <- selected_columns[selected_columns %in% colnames(rt)]
if (length(selected_columns) == 0) {
  stop("No valid column name was selected.")
}

# Select the specified feature column
rt_selected <- rt[, selected_columns, drop = FALSE]

# Ensure that all selected columns are numeric
rt_selected <- as.data.frame(lapply(rt_selected, as.numeric))

# Handle NA values
if (any(is.na(rt_selected))) {
rt_selected <- na.omit(rt_selected)
}

# Calculate the correlation matrix
M <- cor(rt_selected, use = "complete.obs")

# Ensure that the row and column names of the correlation matrix are feature names
rownames(M) <- colnames(rt_selected)
colnames(M) <- colnames(rt_selected)

# Check for invalid values in the correlation matrix
if (any(is.na(M) | is.nan(M) | is.infinite(M))) {
cat("There are invalid values in the correlation matrix.\n")
}

# Handling invalid values
M[is.na(M) | is.nan(M) | is.infinite(M)] <- 0

# Draw and display the correlation heatmap
pheatmap(M, 
         cluster_rows = TRUE, 
         cluster_cols = TRUE, 
         color = colorRampPalette(c("blue", "white", "red"))(50),
         display_numbers = TRUE, 
         fontsize_number = 10, 
         main = "map")

