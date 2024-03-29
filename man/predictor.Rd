% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/trainer.R
\name{predictor}
\alias{predictor}
\title{Model predictions}
\usage{
predictor(obj, test_x, type = NULL)
}
\arguments{
\item{obj}{A fitted model object containing the model parameters..}

\item{test_x}{A numeric matrix for prediction.}

\item{type}{A string specifying prediction method of the model.}
}
\value{
A numeric vector of predicted values for the given data points.
}
\description{
This function predicts new values from a fitted model object for a give set of data points.
The function assumes the model has linear parameters accessible through the object.
}
\examples{
# Data Generation
train_x <- matrix(rnorm(10 * 2), nrow = 10)
test_x <- matrix(rnorm(10 * 2), nrow = 10)
train_y <- rnorm(10)

# Training the model
LRL_objective <- l2_obj(lambda=0.1, loss = 'logistic')
LRL_trainer <- trainer(LRL_objective, SGD_optim(), tol = 1e-5)
result <- LRL_trainer(train_x, train_y, batch_size = 64)

predictor(result, test_x)
}
