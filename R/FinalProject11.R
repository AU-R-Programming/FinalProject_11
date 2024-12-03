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
