#' @title Train a model using a specified objective function and optimizer
#'
#' @description
#' This function trains a model using a given objective function (`objective`),
#' optimizer (`optimizer`), and training data (`train_x`, `train_y`). It supports mini-batch training
#' and early stopping based on convergence criteria.
#'
#' @param objective A function implementing the objective function (cost and gradient) to be optimized.
#' @param optimizer A function implementing the optimizer algorithm to update model parameters.
#' @param tol A numeric value specifying the tolerance for convergence. Defaults to 1e-5.
#' @param epochs An integer specifying the maximum number of training epochs. Defaults to 10000.
#' @param ... Additional arguments passed to the objective function and optimizer.
#'
#' @return A function that fits model parameter and history of costs.
#' ## Arguments
#'   - `train_x` Training data (covariates). It should be a numeric matrix.
#'   - `train_y` Training data (target). It should be a numeric vector.
#'     * For binary classification problem, class should be 1 or -1.
#'   - `batch_size` An integer specifying the number of training examples used in each update step (mini-batch size).
#'     * Set to `"full"` to use the entire dataset in each update.
#'   - `param_init` A numeric vector of initial values for model parameters.
#'     * If not provided, all parameters are initialized to 0.
#'   - `...` Additional arguments passed to the objective function and optimizer.
#' ## Value
#'   - A list containing:
#'     * `params`: Fitted model parameters.
#'     * `cost_history`: History of cost values during training.
#' @export
#' @examples
#' # Data Generation
#' train_x <- matrix(rnorm(10 * 2), nrow = 10)
#' train_y <- rnorm(10)
#'
#' # Train the model#'
#' LRL_objective <- l2_obj(lambda=0.1, loss = 'logistic')
#' LRL_trainer <- trainer(LRL_objective, SGD_optim(), tol = 1e-5)
#' result <- LRL_trainer(train_x, train_y, batch_size = 64)
#' plot(result$cost_history, xlab = 'epoch', ylab = 'cost function', pch=20)
trainer <- function(objective, optimizer, tol = 1e-5, epochs = 10000, ...){
  rval <- function(train_x, train_y, batch_size = 64, params_init = NULL, ...){
    if(is.null(params_init)){
      params = rep(0, ncol(train_x)+1)
    } else {
      params = params_init
    }

    if(!('matrix' %in% class(train_x))) warning("train_x should be matrix!")

    if(batch_size == 'full') batch_size = nrow(train_x)

    iter <- 0
    opt_res <- list(opt_param = NULL)
    obj <- objective(params, train_x, train_y)
    cost <- obj$cost
    cost_history <- cost
    max_step <- nrow(train_x) %/% batch_size
    max_step <- ifelse(nrow(train_x)%%batch_size == 0, max_step, max_step + 1)

    for(epoch in 1:epochs){
      ids <- sample(1:nrow(train_x))
      for(B in 1:max_step){
        iter <- iter + 1
        batch_id <- ((B-1)*batch_size + 1):min(B*batch_size, nrow(train_x))

        if((iter == 1) & batch_size != nrow(train_x)) {
          obj <- objective(params, train_x[batch_id,], train_y[batch_id])
          cost <- obj$cost
        }

        opt_res <- optimizer(params, obj$grad, opt_res$opt_param, iter, epochs * max_step)
        params_new <- opt_res$w
        obj <- objective(params_new, train_x[batch_id,], train_y[batch_id])
        if(batch_size == nrow(train_x)) cost_history <- c(cost_history, obj$cost)

        # update solution
        params <- params_new
        cost <- obj$cost
        if(iter == epochs*max_step) {
          warning("Maximum iteration is reached! Solution may not be converged!")
        }
      }

      if(batch_size != nrow(train_x)){
        obj1 <- objective(params, train_x, train_y)
        cost_history <- c(cost_history, obj1$cost)
      }

      # Convergence Check
      delta <- abs(cost_history[epoch+1] - cost_history[epoch])
      if(delta < tol) break
    }

    invisible(list(params = params, cost_history = cost_history))
  }
}



#' @title Model predictions
#' @description
#' This function predicts new values from a fitted model object for a give set of data points.
#' The function assumes the model has linear parameters accessible through the object.
#' @param obj A fitted model object containing the model parameters..
#' @param test_x A numeric matrix for prediction.
#' @param type A string specifying prediction method of the model.
#' @return A numeric vector of predicted values for the given data points.
#' @export
#'
#' @examples
#' # Data Generation
#' train_x <- matrix(rnorm(10 * 2), nrow = 10)
#' test_x <- matrix(rnorm(10 * 2), nrow = 10)
#' train_y <- rnorm(10)
#'
#' # Training the model
#' LRL_objective <- l2_obj(lambda=0.1, loss = 'logistic')
#' LRL_trainer <- trainer(LRL_objective, SGD_optim(), tol = 1e-5)
#' result <- LRL_trainer(train_x, train_y, batch_size = 64)
#'
#' predictor(result, test_x)
predictor <- function(obj, test_x, type = NULL){
  if(is.vector(test_x)) test_x <- matrix(test_x, nrow=1)
  if(!('matrix' %in% class(test_x))) warning("test_x should be matrix!")
  fhat <- test_x %*% obj$params[-1] + obj$params[1]
  if(!is.null(type) && type == 'binary') fhat <- sign(fhat)
  return(as.numeric(fhat))
}




