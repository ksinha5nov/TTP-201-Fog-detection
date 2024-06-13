# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(ROCR)

# Load the dataset
fog_aware_pivot <- read.csv("fog_aware_stats_horizontal_3_patches_pivot.csv")
fog_aware_pivot$Label_train <- as.integer(fog_aware_pivot$Label == "foggy")

# Function to balance the dataset
balance_dataset <- function(X, y) {
  class_0_indices <- which(y == 0)
  class_1_indices <- which(y == 1)
  
  num_samples_per_class <- min(length(class_0_indices), length(class_1_indices))
  
  sampled_indices_class_0 <- sample(class_0_indices, num_samples_per_class)
  sampled_indices_class_1 <- sample(class_1_indices, num_samples_per_class)
  
  sampled_indices <- c(sampled_indices_class_0, sampled_indices_class_1)
  
  X_balanced <- X[sampled_indices,]
  y_balanced <- y[sampled_indices]
  
  print(table(y_balanced))
  
  return(list(X_balanced, y_balanced))
}

# Columns to keep
to_keep_3_patches <- c('mscn_var_1', 'vertical_var_1', 'sharpness_1', 'coef_or_var_sharpness_1', 'rms_contrast_1')

# Create balanced dataset
X <- fog_aware_pivot[, to_keep_3_patches]
y <- fog_aware_pivot$Label_train

balanced_data <- balance_dataset(X, y)
X_balanced <- balanced_data[[1]]
y_balanced <- balanced_data[[2]]

# Scale the data
preProcValues <- preProcess(X_balanced, method = c("center", "scale"))
X_scaled <- predict(preProcValues, X_balanced)

# Train-test split
set.seed(42)
trainIndex <- createDataPartition(y_balanced, p = 0.75, list = FALSE, times = 1)
X_train <- X_scaled[trainIndex, ]
X_test <- X_scaled[-trainIndex, ]
y_train <- y_balanced[trainIndex]
y_test <- y_balanced[-trainIndex]

# Logistic Regression
model <- svm(x = X_train, y = y_train, kernel = "radial", gamma = 0.1)


summary(model)

# Predictions
y_train_pred <- ifelse(predict(model, type = "response") > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
cat('Train accuracy: ', train_accuracy, '\n')

y_test_pred <- ifelse(predict(model, newdata = data.frame(X_test), type = "response") > 0.5, 1, 0)
test_accuracy <- mean(y_test_pred == y_test)
cat('Test accuracy: ', test_accuracy, '\n')

y_scores <- predict(model, newdata = data.frame(X_test), type = "response")
conf_matrix <- confusionMatrix(factor(y_test_pred), factor(y_test))
confMatrixDf <- as.data.frame(as.table(conf_matrix$table))
# Plot confusion matrix
ggplot(data = confMatrixDf, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1.5, color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix", x = "Predicted Label", y = "True Label") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        plot.title = element_text(hjust = 0.5))
# Convert to factors for posPredValue and sensitivity
y_test_factor <- as.factor(y_test)
y_test_pred_factor <- as.factor(y_test_pred)

# Calculate metrics
accuracy <- mean(y_test_pred == y_test)
precision <- posPredValue(y_test_pred_factor, y_test_factor, positive = "1")
recall <- sensitivity(y_test_pred_factor, y_test_factor, positive = "1")
f1 <- 2 * ((precision * recall) / (precision + recall))
roc_auc <- auc(y_test, y_scores)

cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1, "\n")
cat("AUC-ROC Score: ", roc_auc, "\n")



# Precision-Recall Curve
pr <- prediction(y_scores, y_test)
prf <- performance(pr, measure = "prec", x.measure = "rec")
plot(prf, col = rainbow(7), main = "Precision-Recall Curve")

# ROC Curve
roc <- roc(y_test, y_scores)
plot.roc(roc, col = "darkorange", print.auc = TRUE, main = "ROC Curve")

# Loop through sites
summary_stats <- data.frame(site = character(),
                            Accuracy = numeric(),
                            Precision = numeric(),
                            Recall = numeric(),
                            F1.Score = numeric(),
                            AUC = numeric(),
                            AIC = numeric(),
                            BIC = numeric(),
                            Adjusted_R2 = numeric(),
                            stringsAsFactors = FALSE)

precision_recall <- data.frame(site = character(),
                               precision_array = numeric(),
                               recall_array = numeric(),
                               stringsAsFactors = FALSE)

roc_data <- data.frame(site = character(),
                       fpr = numeric(),
                       tpr = numeric(),
                       stringsAsFactors = FALSE)

y_info <- data.frame(site = character(),
                     y_test = numeric(),
                     y_test_pred = numeric(),
                     y_scores = numeric(),
                     stringsAsFactors = FALSE)

sites <- unique(fog_aware_pivot$site)

for (site in sites) {
  cat(site, "\n")
  fog_aware_site <- fog_aware_pivot %>% filter(site == !!site)
  
  X <- fog_aware_site[, to_keep_3_patches]
  y <- fog_aware_site$Label_train
  
  balanced_data <- balance_dataset(X, y)
  X_balanced <- balanced_data[[1]]
  y_balanced <- balanced_data[[2]]
  
  preProcValues <- preProcess(X_balanced, method = c("center", "scale"))
  X_scaled <- predict(preProcValues, X_balanced)
  
  set.seed(42)
  trainIndex <- createDataPartition(y_balanced, p = 0.75, list = FALSE, times = 1)
  X_train <- X_scaled[trainIndex, ]
  X_test <- X_scaled[-trainIndex, ]
  y_train <- y_balanced[trainIndex]
  y_test <- y_balanced[-trainIndex]
  
  model <- glm(y_train ~ ., data = data.frame(X_train, y_train), family = binomial)
  y_train_pred <- ifelse(predict(model, type = "response") > 0.5, 1, 0)
  y_test_pred <- ifelse(predict(model, newdata = data.frame(X_test), type = "response") > 0.5, 1, 0)
  y_scores <- predict(model, newdata = data.frame(X_test), type = "response")
  
  # Convert to factors for posPredValue and sensitivity
  y_test_factor <- as.factor(y_test)
  y_test_pred_factor <- as.factor(y_test_pred)
  
  accuracy <- mean(y_test_pred == y_test)
  precision <- posPredValue(y_test_pred_factor, y_test_factor, positive = "1")
  recall <- sensitivity(y_test_pred_factor, y_test_factor, positive = "1")
  f1 <- 2 * ((precision * recall) / (precision + recall))
  roc_auc <- auc(y_test, y_scores)
  
  # Calculate AIC, BIC, and Adjusted R²
  aic <- AIC(model)
  bic <- BIC(model)
  n <- length(y_train)
  k <- length(model$coefficients)
  adj_r2 <- 1 - (1 - model$deviance / model$null.deviance) * (n - 1) / (n - k - 1)
  
  cat("Accuracy: ", accuracy, "\n")
  cat("Precision: ", precision, "\n")
  cat("Recall: ", recall, "\n")
  cat("F1 Score: ", f1, "\n")
  cat("AUC-ROC Score: ", roc_auc, "\n")
  cat("AIC: ", aic, "\n")
  cat("BIC: ", bic, "\n")
  cat("Adjusted R²: ", adj_r2, "\n")
  
  pr <- prediction(y_scores, y_test)
  prf <- performance(pr, measure = "prec", x.measure = "rec")
  precision_array <- prf@y.values[[1]]
  recall_array <- prf@x.values[[1]]
  
  pr_roc <- performance(pr, measure = "tpr", x.measure = "fpr")
  fpr <- pr_roc@x.values[[1]]
  tpr <- pr_roc@y.values[[1]]
  
  summary_stats <- rbind(summary_stats, data.frame(site = site,
                                                   Accuracy = accuracy,
                                                   Precision = precision,
                                                   Recall = recall,
                                                   F1.Score = f1,
                                                   AUC = roc_auc,
                                                   AIC = aic,
                                                   BIC = bic,
                                                   Adjusted_R2 = adj_r2))
  
  precision_recall <- rbind(precision_recall, data.frame(site = site,
                                                         precision_array = precision_array,
                                                         recall_array = recall_array))
  
  roc_data <- rbind(roc_data, data.frame(site = site, fpr = fpr, tpr = tpr))
  y_info <- rbind(y_info, data.frame(site = site, y_test = y_test, y_test_pred = y_test_pred, y_scores = y_scores))
}

# Print summary statistics
print(summary_stats)

# Plot Precision-Recall Curve
ggplot(precision_recall, aes(x = recall_array, y = precision_array, color = site)) +
  geom_line() +
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision") +
  theme_minimal()

# Plot ROC Curve
ggplot(roc_data, aes(x = fpr, y = tpr, color = site)) +
  geom_line() +
  geom_abline(linetype = "dashed") +
  labs(title = "ROC Curve", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()


