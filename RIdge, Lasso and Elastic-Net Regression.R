install.packages("glmnet")
library(glmnet)

set.seed(42)

n <- 1000
p <- 5000 # number of parameters to estimate
real_p <- 15 # the remaining parameters are noises

x <- matrix(rnorm(n*p), nrow=n, ncol=p)
# 5000000 values in the matrix come from a standard normal distribution
# (with mean = 0 and sd = 1)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)
# a vector of values that we'll predict with the data in x
# apply() returns a vector of 1000 values that are the sums of the
# first 15 columns in x, since x has 1000 rows
# x[,1:real_p]: isolates columns 1 through 15 from x
# 1: specifies we wanna perform a function on each row of data that we've isolated from x
# sum: the function we wanna apply to each row
# add a little noise using rnorm(), which returns 1000 random values from
# a standard normal distribution

train_rows <- sample(1:n, .66*n)
# sample() randomly selects numbers between 1 and n (no of rows in the dataset),
# and selects 0.66 * n row numbers, ie 2/3 of the data will be in the training set
x.train <- x[train_rows, ]
x.test <- x[-train_rows, ] # the remaining rows

y.train <- y[train_rows]
y.test <- y[-train_rows]

### Ridge Regression
# fit a model to the training data
alpha0.fit <- cv.glmnet(x.train, y.train, type.measure="mse",
  alpha=0, family="gaussian")
# cv: cross validation to obtain the optimal values for lambda
# cv.glmnet() uses 10-fold cv by default
# use x.train to predict y.train
# Unlike lm() or glm(), cv.glmnet() doesn't accept formula notation
# x and y must be passed in separately
# type.measure: how the cv will be evaluated
# "mse": mean squared error (sum of the squared residuals divided by the sample size)
# use "deviance" if we were applying Elastic-Net Regression to Logistic Regression
# alpha=0 since we are starting with Ridge Regression
# family="gaussian": teels glmnet we are doing Linear Regression

# Apply alpha0.fit to the testing data
alpha0.predicted <- predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)
# s: size of the penalty, set to one of the optimal values for lambda stored in alpha0.fit
# lambda.1se: the value for lambda, stored in alpha0.fit, that resulted in the simplest model
# (ie the model with the fewest non-zero parameters) and wwas within 1 standard error of the lambda
# that had the smallest sum
# lambda.min: the lambda that resulted in the smallest sum
# lambda.1se is indistinguishable from lambda.min,
# but it results in a model with fewer parameters.
mean((y.test - alpha0.predicted)^2)

alpha1.fit <- cv.glmnet(x.train, y.train, type.measure="mse",
  alpha=1, family="gaussian")
alpha1.predicted <- predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)
mean((y.test - alpha1.predicted)^2)
# Lasso Regression is better for this dataset

alpha0.5.fit <- cv.glmnet(x.train, y.train, type.measure="mse",
  alpha=0.5, family="gaussian")
alpha0.5.predicted <- predict(alpha0.5.fit, s=alpha0.5.fit$lambda.1se, newx=x.test)
mean((y.test - alpha0.5.predicted)^2)

# try different values of alpha
list.of.fits <- list()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  # fit.name: alpha0, alpha0.1, ...
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure="mse", alpha=i/10,
      family="gaussian")
}

results <- data.frame()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)

  predicted <-
    predict(list.of.fits[[fit.name]],
      s=list.of.fits[[fit.name]]$lambda.1se, newx=x.test)

  mse <- mean((y.test - predicted)^2)

  temp <- data.frame(alpha=i/10, mse=mse, fit.name=fit.name)
  # append temp to bottom row of the results data.frame
  results <- rbind(results, temp)
}
results
# the mse are slightly different from what we got before
# as the parameter values, prior to regularisation and optimisation,
# are randomly initialised

# Elastic-Net Regression is the best method to use with this data
# with alpha=0.9