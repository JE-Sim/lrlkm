#' @title Stochastic Gradient Descent with momentum
#' @description
#' This function implements the Stochastic Gradient Descent (SGD) optimizer with momentum
#' and optional Nesterov accelerated gradients and learning rate decay.
#' @param learning_rate A numeric value specifying the initial learning rate. Defaults to 0.1.
#' @param momentum A numeric value between 0 and 1 controlling the momentum term. Defaults to 0.
#' @param nesterov A logical value indicating whether to use Nesterov accelerated gradients. Defaults to `FALSE`.
#' @param decay A character string specifying the learning rate decay schedule:
#'   * "constant" (default): No decay.
#'   * "linear": Linear decay over iterations.
#'   * "inv_sqrt": Inverse square root decay.
#'   * "cosine": Cosine annealing decay.
#'
#' @return A function performing the SGD update step.
#' @export
#'
#' @examples
#' # Example 1: SGD with constant learning rate and momentum
#' optimizer <- SGD_optim(learning_rate = 0.01, momentum = 0.9)
#'
#' # Example 2: SGD with linear learning rate decay
#' optimizer <- SGD_optim(learning_rate = 0.1, decay = "linear")
#'
#' # ... similar examples for other decay options
SGD_optim <- function(learning_rate = 0.1, momentum = 0, nesterov = FALSE, decay = 'constant'){
  if(momentum == 0) v <- 0

  rval <- function(w, g, optim_param=NULL, ...){
    if(decay == 'linear'){
      lr <- learning_rate*(1 - iter/max_iter)
    } else if(decay == 'inv_sqrt'){
      lr <- learning_rate/sqrt(iter)
    } else if(decay == 'cosine'){
      lr <- 1/2 * learning_rate * (1 + cos(iter*pi/max_iter))
    } else {
      lr <- learning_rate
    }

    if(!is.null(optim_param)) v <- optim_param$v

    v <- momentum * v - lr * g # velocity
    w <- w + v # update
    if(is.logical(nesterov) & nesterov){
      w <- w + momentum * v # Nesterov accelerated gradient
    }
    return(list(w = w, optim_param = list(v = v)))
  }
}

#' @title Adaptive Moment Estimation (Adam) optimizer
#' @description
#' This function implements the Adam optimizer, an adaptive learning rate optimization algorithm
#' that estimates first and second moments of the gradient to adjust learning rates.
#' @param learning_rate A numeric value specifying the initial learning rate. Defaults to 0.001.
#' @param beta1 A numeric value between 0 and 1 controlling the exponential decay rate for the first moment estimate. Defaults to 0.9
#' @param beta2 A numeric value between 0 and 1 controlling the exponential decay rate for the second moment estimate. Defaults to 0.999
#' @param eps A small numeric value for numerical stability. Defaults to 1e-8.
#'
#' @return A function performing the Adam update step.
#'
#' @export
#'
#' @examples
#' # Example usage
#' optimizer <- ADAM_optim(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999)
#' # ... use the optimizer within your training loop
ADAM_optim <- function(learning_rate = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8){
  m <- 0; v <- 0
  rval <- function(w, g, optim_param=NULL, iter, ...){
    if(!is.null(optim_param)){
      m <- optim_param$m
      v <- optim_param$v
    }
    m <- beta1 * m + (1 - beta1) * g
    v <- beta2 * v + (1 - beta2) * (g^2)
    m_hat <- m / (1 - beta1^iter) # first moment estimate
    v_hat <- v / (1 - beta2^iter) # second moment estimate
    w <- w - learning_rate * m_hat / (sqrt(v_hat) + eps)
    return(list(w = w, optim_param = list(m = m, v = v)))
  }
}
