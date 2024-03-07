#' @title Empirical kernel mapping matrix
#' @description
#' This function generates an empirical kernel mapping function based on the provided data and kernel.
#' The function uses a subset of training data to approximate the full kernel matrix, improving computational efficiency.
#' @param train_x Training data (covariates). It should be a numeric matrix.
#' @param landmarks A set of indices from `train_x` to use as landmarks. It should be a numeric vector.
#' @param kernel A kernel generating function provided in the `kernlab` package.
#' @param q Cumulative proportion of eigenvalues to retain (for dimensionality reduction). Defaults to `NULL` (use full rank). It should be a numeric value between 0 and 1.
#' @param scale A logical value indicating whether to standardize the data. Defaults to `TRUE`.
#' @return A function generates empirical kernel mapping matrix of \eqn{k(}`train_x`, `test_x`\eqn{)}.
#' ## Arguments
#'   - `test_x` Test data (covariates). It should be a numeric matrix.
#' ## Value
#'   - Empirical kernel mapping matrix as a numeric matrix.
#'
#' @export
#' @examples
#' library(kernlab)
#' rbfkernel <- rbfdot(sigma = 0.1)
#'
#' # Fit empirical kernel map
#' data(iris)
#' landmarks <- sample(1:100, 10)
#' emp_kernel <- emp_kernel_map(iris[1:100, 1:4], landmarks, rbfkernel, q=.95)
#'
#' # Empirical kernel mapping matrices
#' F_train <- emp_kernel() # empirical kernel mapping for training data
#' F_test <- emp_kernel(iris[101:150, 1:4]) # empirical kernel mapping for test data
emp_kernel_map <- function(train_x, landmarks, kernel, q = NULL, scale = T){
  if(scale){
    scaler <- standardScaler(train_x)
    train_x <- as.matrix(scaler())
  }

  K <- kernlab::kernelMatrix(kernel, train_x, train_x[landmarks,])
  evd <- svd(K[landmarks,])
  sel <- ifelse(is.null(q), qr(K[landmarks,])$rank, min(sum(cumsum(evd$d)/sum(evd$d) < q)+1, length(landmarks)))
  M <- tcrossprod(evd$u[,1:sel], diag(sel)*(1/sqrt(evd$d[1:sel])))
  rval <- function(test_x = NULL){
    if(scale) test_x <- as.matrix(scaler(test_x))
    if(is.null(test_x)) K <- kernlab::kernelMatrix(kernel, test_x, train_x[landmarks,])
    return(K %*% M)
  }
}
