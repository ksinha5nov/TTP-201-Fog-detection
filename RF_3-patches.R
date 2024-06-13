# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(e1071)
library(gplots)

# Load the data
df <- read.csv("E:/TTP 201/Project/fog_aware_stats_horizontal_3_patches_pivot.csv")

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

# Print the model structure and parameters
print(model)

# Function to calculate permutation importance
calculate_permutation_importance <- function(model, X_test, y_test) {
  baseline_accuracy <- mean(predict(model, X_test) == y_test)
  importance <- numeric(ncol(X_test))
  
  for (i in seq_along(importance)) {
    X_permuted <- X_test
    X_permuted[, i] <- sample(X_permuted[, i])
    permuted_accuracy <- mean(predict(model, X_permuted) == y_test)
    importance[i] <- baseline_accuracy - permuted_accuracy
  }
  
  importance
}

# Calculate permutation importance
perm_importance <- calculate_permutation_importance(model, X_test, y_test)
perm_importance_df <- data.frame(Feature = colnames(X_test), Importance = perm_importance)

# Plot permutation importance
ggplot(perm_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Feature") +
  ylab("Permutation Importance") +
  ggtitle("Permutation Importance for Random Forest Model") +
  theme_minimal()


# For a more detailed summary, including confusion matrix from cross-validation
print(cv_results)

# Make predictions on the test set
y_pred_test <- predict(model, X_test)

# Calculate metrics
conf_matrix <- confusionMatrix(y_pred_test, as.factor(y_test))
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
f1 <- 2 * (precision * recall) / (precision + recall)

roc_auc <- roc(as.numeric(y_test), as.numeric(as.factor(y_pred_test)))$auc

n <- nrow(X_test)
p <- ncol(X_test)
rss <- sum((y_test - as.numeric(as.factor(y_pred_test)))^2)
tss <- sum((y_test - mean(y_test))^2)
r2 <- 1 - rss/tss
adj_r2 <- 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

# Print the metrics
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1))
print(paste("AUC-ROC Score:", roc_auc))
print(paste("Adjusted R-squared:", adj_r2))

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
            main = paste("Confusion Matrix for", model_name, "3 patches (using Test Set)"), 
            xlab = "Predicted label", ylab = "True label")
}

plot_confusion_matrix(y_test, y_pred_test, "Random Forest Classifier")

library(PRROC)

# Get prediction probabilities
y_scores <- predict(model, X_test, type = "prob")[, 2]

# Calculate precision and recall for various thresholds
pr <- pr.curve(scores.class0 = y_scores, weights.class0 = y_test, curve = TRUE)

# Plot precision-recall curve
plot(pr$curve[, 1], pr$curve[, 2], type = "l", col = "red", lwd = 2, main = "Precision-Recall Curve for Random Forest 3 patches", xlab = "Recall", ylab = "Precision")

# Calculate ROC curve and AUC
roc_obj <- roc(y_test, y_scores)
roc_auc <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, col = "darkorange", lwd = 2, main = sprintf("ROC curve 3 patches (AUC = %.2f)", roc_auc))
abline(a = 0, b = 1, col = "navy", lwd = 2, lty = 2)