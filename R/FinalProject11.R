## work here

get_predicted_prob <- function(beta, X) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(p)
}

# X being a matrix of predictors where each row is an observation
# y being a n x 1 column of responses where each row contains a response
# The arguments passed are at the ith observation to be used in the optimization
# function
loss_fn <- function(beta, X_i, y_i) {
  p_i <- get_predicted_prob(beta, X)
  loss <- sum(-y_i * log(p_i) - (1 - y_i) * log(1 - p_i))
  return(loss)
}

get_initial_beta <- function(X, y) {
  solve(t(X) %*% X) %*% t(X) %*% y
}

optimization_fn <- function(X, y) {

  n <- nrow(X)
  # add intercept to data
  # TODO Replace with model.matrix()
  design <- model.matrix(~., X)
  initial_beta <- get_initial_beta(design, y)

  result <- optim(par = initial_beta, fn = loss_fn)

  beta_hat <- result$par
  return(beta_hat)
}


## work here




bootstrapCI <- function(X, y, alpha = 0.05, B = 20) {
  n <- nrow(X)  # Number of observations
  X <- cbind(1, X)  # Add intercept column
  p <- ncol(X)  # Number of predictors (including intercept)

  # Matrix to store beta estimates for each bootstrap iteration
  beta_hat <- matrix(0, nrow = B, ncol = p)

  for (b in 1:B) {
    # Resample the data with replacement
    sample_indices <- sample(1:n, n, replace = TRUE)
    X_boot <- X[sample_indices, ]
    y_boot <- y[sample_indices]

    # Use the provided optimization function to estimate beta coefficients
    beta_hat[b, ] <- optimization_fn(X_boot, y_boot)
  }

  # Calculate confidence intervals for each coefficient
  CI <- matrix(0, nrow = p, ncol = 2)
  for (j in 1:p) {
    CI[j, ] <- quantile(beta_hat[, j], probs = c(alpha / 2, 1 - alpha / 2))
  }

  # Add labels to the output
  colnames(CI) <- c("Lower Bound", "Upper Bound")
  rownames(CI) <- colnames(X)

  return(CI)
}





## https://chatgpt.com/share/674fbe0f-bb04-8009-a85f-4bd1ca6ae0e2










## work here
compute_metrics <- function(predicted_probs, y, cutoff = 0.5) {
  # Convert predicted probabilities to predicted class labels using the cutoff
  predictions <- ifelse(predicted_probs > cutoff, 1, 0)

  # Confusion Matrix: Predicted vs Actual
  confusion_matrix <- table(Predicted = predictions, Actual = y)

  # Ensure the confusion matrix has all 4 possible entries (0/1 for Predicted and Actual)
  if (length(confusion_matrix) == 1) {
    confusion_matrix <- matrix(c(0, 0, 0, 0), nrow = 2, dimnames = list(Predicted = c(0, 1), Actual = c(0, 1)))
  }

  # Rename for easier reference
  true_pos <- confusion_matrix[2, 2]  # Predicted = 1, Actual = 1
  false_pos <- confusion_matrix[2, 1]  # Predicted = 1, Actual = 0
  false_neg <- confusion_matrix[1, 2]  # Predicted = 0, Actual = 1
  true_neg <- confusion_matrix[1, 1]   # Predicted = 0, Actual = 0

  # Prevalence (proportion of actual positives)
  prevalence <- (true_pos + false_neg) / sum(confusion_matrix)

  # Accuracy (proportion of correct predictions)
  accuracy <- (true_pos + true_neg) / sum(confusion_matrix)

  # Sensitivity (True Positive Rate)
  sensitivity <- true_pos / (true_pos + false_neg)

  # Specificity (True Negative Rate)
  specificity <- true_neg / (true_neg + false_pos)

  # False Discovery Rate (FDR)
  false_discovery_rate <- false_pos / (true_pos + false_pos)

  # Diagnostic Odds Ratio (DOR)
  diagnostic_odds_ratio <- (true_pos * true_neg) / (false_pos * false_neg)

  # Return all metrics in a list
  metrics <- list(
    Confusion_Matrix = confusion_matrix,
    Prevalence = prevalence,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    False_Discovery_Rate = false_discovery_rate,
    Diagnostic_Odds_Ratio = diagnostic_odds_ratio
  )

  return(metrics)
}

## https://chatgpt.com/share/674fc2c8-0348-8001-89a8-838a83fcc41f
