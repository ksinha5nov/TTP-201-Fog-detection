# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(e1071)
library(gplots)

# Load the data
df <- read.csv("E:/TTP 201/Project/fog_aware_stats_4_equal_patches_pivot.csv")

# Create the Label_train column
df$Label_train <- ifelse(df$Label == 'foggy', 1, 0)

# Check value counts
table(df$Label_train)
table(df$Label)

# Number of data points to drop from the "foggy" class
num_to_drop <- 4752

# Check if the number to drop is greater than the count of "foggy" class
label_counts <- table(df$Label)

if (num_to_drop > label_counts["not_foggy"]) {
  print("Warning: The specified number of data points to drop is greater than the count of 'foggy' class.")
} else {
  # Randomly drop data points from the "foggy" class
  set.seed(42)
  foggy_indices <- which(df$Label_train == 0)
  df_to_drop <- df[foggy_indices[sample(length(foggy_indices), num_to_drop)], ]
  df_filtered <- df[-as.numeric(rownames(df_to_drop)), ]
  
  # Display the resulting DataFrame
  print(df_filtered)
  print(label_counts)
}

label_counts <- table(df_filtered$Label)
print(label_counts)

# Prepare data for modeling
X <- df_filtered %>% select(-Label, -Label_train, -photo, -site)
X <- as.data.frame(scale(X))
y <- df_filtered$Label_train

set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Create a RandomForestClassifier
model <- randomForest(X_train, as.factor(y_train), ntree = 100, mtry = sqrt(ncol(X_train)), maxnodes = 10, random_state = 42)

# Perform cross-validation predictions
cv_results <- train(X_train, as.factor(y_train), method = "rf", trControl = trainControl(method = "cv", number = 5))

# Make predictions on the test set
y_pred_test <- predict(model, X_test)

# Calculate metrics
conf_matrix <- confusionMatrix(y_pred_test, as.factor(y_test))
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
f1 <- 2 * (precision * recall) / (precision + recall)

roc_auc <- roc(as.numeric(y_test), as.numeric(as.factor(y_pred_test)))$auc

# Print the metrics
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1))
print(paste("AUC-ROC Score:", roc_auc))

# Plot confusion matrix
plot_confusion_matrix <- function(y_test, y_test_pred, model_name) {
  conf_matrix <- confusionMatrix(as.factor(y_test_pred), as.factor(y_test))
  cm <- conf_matrix$table
  
  group_names <- c('True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)')
  group_counts <- as.character(cm)
  group_percentages <- round(prop.table(cm), 2)
  
  labels <- paste(group_names, group_counts, group_percentages, sep = "\n")
  labels <- matrix(labels, nrow = 2, ncol = 2)
  
  custom_colors <- colorRampPalette(c("lightblue", "darkblue"))(100)
  
  heatmap.2(as.matrix(cm), cellnote = labels, notecol = "black", density.info = "none", trace = "none", 
            col = custom_colors, margins = c(5, 5), dendrogram = "none", key = FALSE, 
            main = paste("Confusion Matrix for", model_name, "4 patches (using Test Set)"), 
            xlab = "Predicted label", ylab = "True label")
}

plot_confusion_matrix(y_test, y_pred_test, "Random Forest Classifier")

library(PRROC)

# Get prediction probabilities
y_scores <- predict(model, X_test, type = "prob")[, 2]

# Calculate precision and recall for various thresholds
pr <- pr.curve(scores.class0 = y_scores, weights.class0 = y_test, curve = TRUE)

# Plot precision-recall curve
plot(pr$curve[, 1], pr$curve[, 2], type = "l", col = "red", lwd = 2, main = "Precision-Recall Curve for Random Forest 4 patches", xlab = "Recall", ylab = "Precision")

# Calculate ROC curve and AUC
roc_obj <- roc(y_test, y_scores)
roc_auc <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, col = "darkorange", lwd = 2, main = sprintf("ROC curve 4 patches (AUC = %.2f)", roc_auc))
abline(a = 0, b = 1, col = "navy", lwd = 2, lty = 2)