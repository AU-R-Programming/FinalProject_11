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







## work here

predicted_probs<-1/1+exp(-design %*% beta_optimized)
predictions<- ifelse(predicted_probs>0.5,1,0)
confusion_matrix<-table(Predicted=predicted_probs, Actual=y)
print(confusion_matrix)


#rename to make it easier
true_pos<-confusion_matrix[1,1]
false_pos<-confusion_matrix[1,2]
false_neg<-confusion_matrix[2,1]
true_neg<-confusion_matrix[2,2]

#Prevalence
prevalence <- (true_pos+false_neg)/sum(confusion_matrix)
print(prevalence)

#Accuracy
accuracy <- (true_pos+true_neg)/sum(confusion_matrix)
print(accuracy)

#Sensitivity
sensitivity<-true_pos/(true_pos+false_neg)
print(sensitivity)

#Specificity
specificity <- true_neg/(true_neg+false_pos)
print(specificity)

#False Discovery Rate
false_discovery_rate <- false_pos/(true_pos+false_pos)
print(false_discovery_rate)

#Diagnostic Odds Ratio
diagnostic_odds_ratio<-(true_pos+true_neg)/(false_pos+false_neg)
print(diagnostic_odds_ratio)




