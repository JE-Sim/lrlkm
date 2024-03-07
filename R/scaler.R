#' @title Standardized scaler
#' @description
#' This function is a scaler that standardizes the columns of a numeric matrix based on centers and scalings of the provided data.
#' Standardization involves centering the data (subtracting the mean) and scaling the data (dividing by the standard deviation).
#' @param data The base data to be used for scaling. It should be a numeric matrix.
#' @param center A logical value indicating whether to center the data. Defaults to `TRUE`.
#' @param scale A logical value indicating whether to scale the data. Defaults to `TRUE`.
#' @return A function that performs standardization based on the provided data.
#' ## Arguments
#'   - `x` The new data to be standardized. It should be a numeric matrix. Defaults to `data`.
#' ## Value
#'   - The standardized data as a numeric matrix.
#' @export
#' @examples
#' # Fit the scaler
#' data(iris)
#' scaler <- standardScaler(iris[1:100, 1:4])
#'
#' # Standardized base data
#' scaler()
#'
#' # Scale new data
#' scaler(iris[101:150, 1:4])
standardScaler <- function(data, center = T, scale = T) {
  if(is.logical(center) && center){
    center <- colMeans(data)
  }

  if(is.logical(scale) && scale){
    scale <- apply(data, 2, sd)
  }

  rval <- function(x = NULL){
    if(is.null(x)) x <- data
    return(scale(x, center, scale))
  }
}

