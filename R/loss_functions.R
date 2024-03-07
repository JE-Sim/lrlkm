#' @title Compute input of loss function and gradients
#' @description
#' This function defines function of the core components of various loss functions,
#' including both the loss input (`u`) and its gradient(`du`).
#' It currently supports regression ('reg', 'regression') and and binary classification ('binary') problem.
#' The specific components differ depending on the chosen `type`.
#' @param type A character string specifying the type of input function.
#'   - 'reg' or 'regression' : Used for regression problem.
#'   - 'binary' : Used for binary classification problem.
#' @return A function computes input functions and their gradients based on the provided loss type.
#' ## Arguments
#'   - `train_y` A numeric vector of true target values.
#'   - `f_hat` A numeric vector of decision function value
#' ## Value
#'   - A named list containing `u` (loss input) and `du` (gradient of the loss input)
#' @export
#' @examples
#' # Regression example
#' reg_input_func <- loss_components("reg")
#' reg_loss_input <- reg_input_func(train_y = rnorm(10), f_hat = rnorm(10))
#' reg_loss_input$u # input value of loss function
#' reg_loss_input$du # Access gradients
#'
#' # Binary classification example
#' bin_input_func <- loss_components("binary")
#' bin_loss_input <- bin_input_func(train_y = rbinom(10, 1, 0.5), f_hat = rnorm(10))
#' bin_loss_input$u # input value of loss function
#' bin_loss_input$du # Access gradients
loss_components <- function(type){
  rval <- function(train_y, f_hat){
    if(type %in% c('reg', 'regression')){
      return(list(u = train_y - f_hat, du = -1))
    } else if(type == 'binary'){
      return(list(u = train_y * f_hat, du = train_y))
    } else {
      stop("type of inputs should be one of 'reg', 'regression' or 'binary'!")
    }
  }
}


#' @title Compute Squared loss and its gradient
#' @description
#' This function defines the function of the squared loss and its gradient.
#' The squared loss is commonly used as a loss function in regression problems.
#' @return A function performs value (`loss`) and its gradients (`grad`) of the squared loss function.
#' @export
#'
#' @examples
#' u <- rnorm(10)
#' loss <- squared_loss()
#' loss_result <- loss(u)
#' loss_result$loss # loss value
#' loss_result$grad # gradients
squared_loss <- function(){
  rval <- function(u){
    loss <- u^2
    grad <- 2*u
    return(list(loss = loss, grad = grad))
  }
}


#' @title Compute Check loss and its gradient
#' @description
#' This function defines the function of the check loss and its gradient for a given value `tau`.
#' The check loss is a robust loss function that is commonly used as a loss function in quantile regression problem.
#' @param tau A numeric value between 0 and 1 specifying the quantile to estimate. Defaults to 0.5.
#' @return A function performs value (`loss`) and its gradients (`grad`) of the `tau`-th quantile loss function.
#' @export
#'
#' @examples
#' u <- rnorm(10)
#' loss <- check_loss()
#' loss_result <- loss(u)
#' loss_result$loss # loss value
#' loss_result$grad # gradients
check_loss <- function(tau = 0.5){
  rval <- function(u){
    loss <- u * (tau - (u < 0))
    grad <- tau - (u < 0)
    return(list(loss = loss, grad = grad))
  }
}


#' @title Compute Epsilon-insensitive loss and its gradient
#' @description
#' This function defines the function of the epsilon-insensitive loss and its gradient for a given value `eps`.
#' The epsilon insensitive loss is commonly used as a loss function in support vector regression problems.
#' It ignores errors smaller than a specified epsilon value (`eps`).
#' @param eps A positive numeric value specifying the threshold for the epsilon-insensitive region. Defaults to 0.1.
#' @return A function performs value (`loss`) and its gradients (`grad`) of the epsilon insensitive loss function with `eps` threshold.
#' @export
#'
#' @examples
#' u <- rnorm(10)
#' loss <- eps_insensitive_loss(0.5)
#' loss_result <- loss(u)
#' loss_result$loss # loss value
#' loss_result$grad # gradients
eps_insensitive_loss <- function(eps = 0.1){
  rval <- function(u){
    loss <- pmax(abs(u) - eps, 0)
    grad <- (u-eps > 0) - (u+eps < 0)
    return(list(loss = loss, grad = grad))
  }
}

#' @title Compute Hinge loss and its gradient
#' @description
#' This function defines the function of the hinge loss and its gradient.
#' The hinge loss is commonly used as a loss function in support vector machine (SVM) classification problems.
#' @param s A numeric value specifiying the margin parameter. Defaults to 1.
#' @return A function performs value (`loss`) and its gradients (`grad`) of the hinge loss function.
#' @export
#'
#' @examples
#' u <- rnorm(10)
#' loss <- hinge_loss()
#' loss_result <- loss(u)
#' loss_result$loss # loss value
#' loss_result$grad # gradients
hinge_loss <- function(s = 1){
  rval <- function(u){
    loss <- pmax(s-u, 0)
    grad <- -(s-u > 0)
    return(list(loss = loss, grad = grad))
  }
}

#' @title Compute Logistic loss and its gradient
#' @description
#' This function defines the function of the logistic loss and its gradient.
#' The logistic loss is commonly used as a loss function in logistic regression,
#' measuring the difference between the predicted probability and the true label.
#' @return A function performs value (`loss`) and its gradients (`grad`) of the logistic loss function.
#' @export
#'
#' @examples
#' u <- rnorm(10)
#' loss <- logistic_loss()
#' loss_result <- loss(u)
#' loss_result$loss # loss value
#' loss_result$grad # gradients
logistic_loss <- function(){
  rval <- function(u){
    loss <- log(1+exp(-u))
    grad <- -1/(1+exp(u))
    return(list(loss = loss, grad = grad))
  }
}

