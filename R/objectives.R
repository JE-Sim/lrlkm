#' @title Objective function with L2 regularization for various loss functions
#' @description
#' This function implements the objective function for L2 regularized linear models with various loss functions.
#' It computes the objective function value (cost) and its gradient for a given set of parameters,
#' training data, and a specified loss function.
#' @param loss A character string specifying the loss function to use. Supported options include:
#'   * "ridge": ridge regression (squared loss)
#'   * "qr": quantile regression
#'   * "svr": support vector regression
#'   * "svm": support vector machine (binary classification)
#'   * "lr": logistic regression
#' @param lambda A numeric value specifying the L2 regularization penalty coefficient.
#' @param loss_input A character string or function specifying the format of the data for the loss function.
#'   * For regression problems ("ridge", "qr", "svr"), set to "reg" (default).
#'   * For classification problems ("svm", "lr"), set to "binary" (default).
#' @param ... Additional arguments passed to the specified loss function.
#'   * For "qr": `tau` (quantile parameter, defaults to 0.5).
#'   * For "svr": `eps` (epsilon parameter, defaults to 0.1).
#'
#' @return A function that computes value of objective function (`cost`) and its gradient (`grad`)
#' ## Arguments
#'   - `params` A numeric vector specifying a bias and weights of the linear model.
#'   - `train_x` Training data (covariates). It should be a numeric matrix.
#'   - `train_y` Training data (target). It should be a numeric vector.
#'     * For binary classification problem, class should be 1 or -1.
#' ## Value
#'  - A list containing:
#'    * `cost`: The value of the objective function (cost).
#'    * `grad`: The gradient of the objective function with respect to the parameters.
#'
#' @export
#'
#' @examples
#' # Example 1: Ridge regression
#' ridge_obj <- l2_obj("ridge", lambda = 0.1)
#' train_x <- matrix(rnorm(10 * 2), nrow = 10)
#' train_y <- rnorm(10)
#' params <- c(0, rep(1, 2))  # Bias and weights
#' result <- ridge_obj(params, train_x, train_y)
#' cost <- result$cost
#' grad <- result$grad
#'
#' # Example 2: Quantile regression with tau = 0.7
#' qr_obj <- l2_obj("qr", lambda = 0.1, tau = 0.7)
#' train_x <- matrix(rnorm(10 * 2), nrow = 10)
#' train_y <- rnorm(10)
#' params <- c(0, rep(1, 2))
#' result <- qr_obj(params, train_x, train_y)
#' cost <- result$cost
#' grad <- result$grad
#'
#' # ... similar examples for other supported loss functions
l2_obj <- function(loss, lambda = 0.1, loss_input = NULL, ...){
  if(is.character(loss_input)) loss_input <- loss_components(loss_input)
  args <- list(...)

  if(loss == 'ridge'){
    # ridge regression
    loss <- squared_loss()
    if(is.null(loss_input)) loss_input <- loss_components('reg')
  } else if(loss %in% c('qr', 'quantile')){
    # quantile regression
    tau <- ifelse('tau' %in% names(args), args$tau, 0.5)
    loss <- check_loss(tau)
    if(is.null(loss_input)) loss_input <- loss_components('reg')
  } else if(loss %in% c('svr', 'eps-svr')){
    # Support Vector Regression
    eps <- ifelse('eps' %in% names(args), args$eps, 0.1)
    loss <- eps_insensitive_loss(eps)
    if(is.null(loss_input)) loss_input <- loss_components('reg')
  } else if(loss %in% c('svm', 'svc', 'C-svc')){
    # Support Vector Machine for binary Classification
    # Y %in% c(1, -1)
    loss <- hinge_loss()
    if(is.null(loss_input)) loss_input <- loss_components('binary')
  } else if(loss %in% c('lr', 'logistic')){
    # Logistic regression
    # Y %in% c(1, -1)
    loss <- logistic_loss()
    if(is.null(loss_input)) loss_input <- loss_components('binary')
  }

  if(is.null(loss_input)) warning("unused arguments (loss_input). Proper loss_input is required!")

  rval <- function(params, train_x, train_y){
    if(is.vector(train_x)) train_x <- matrix(train_x, nrow=1)

    f_hat <- train_x %*% params[-1]
    u <- loss_input(train_y, f_hat + params[1])
    l <- loss(u$u)
    grad <- c(0, lambda * params[-1]) + as.numeric(crossprod(u$du * l$grad, cbind(1, train_x)))/nrow(train_x)
    cost <- as.numeric(lambda/2 * crossprod(params[-1]) + mean(l$loss))
    return(list(cost = cost, grad = grad))
  }
}
