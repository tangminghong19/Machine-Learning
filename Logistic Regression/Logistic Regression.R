url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" # nolint
data <- read.csv(url, header = FALSE)
head(data)
colnames(data) <- c(
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
)
head(data)
str(data)

data[data == "?"] <- NA

data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <- as.factor(data$sex)

data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

data$ca <- as.integer(data$ca)
data$ca <- as.factor(data$ca)

data$thal <- as.integer(data$thal)
data$thal <- as.factor(data$thal)

data$num <- ifelse(test = data$num == 0, yes = "Healthy", no = "Unhealthy")
data$num <- as.factor(data$num)

str(data)

nrow(data[is.na(data$ca) | is.na(data$thal),])
data[is.na(data$ca) | is.na(data$thal),]

nrow(data)

# remove rows with missing data from the dataset
data <- data[!(is.na(data$ca) | is.na(data$thal)),]
nrow(data)

xtabs(~ num + sex, data = data)
xtabs(~ num + cp, data = data)
xtabs(~ num + fbs, data = data)

xtabs(~ num + restecg, data = data)
##            restecg
## num          0  1  2
##   Healthy   92  1 67
##   Unhealthy 55  3 79
# Only 4 patients represent level 1
# This could get in the way of finding the best fitted line

xtabs(~ num + exang, data = data)
xtabs(~ num + slope, data = data)
xtabs(~ num + ca, data = data)
xtabs(~ num + thal, data = data)

# binomial: Logistic Regression
logistic <- glm(num ~ sex, data = data, family = "binomial")
summary(logistic)

## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)
## (Intercept)  -1.0438     0.2326  -4.488 7.18e-06 ***
## sexM          1.2737     0.2725   4.674 2.95e-06 ***
# correspond to the following model:
# log(odds) of having hd = -1.0438 + 1.2737 * the patient is male
# sexM = 0 when the patient is female, = 1 when the patient is male
# Std. Error and z value shows how the Wald's test was computed for both coeff.
# Both p-values < 0.05, they're both statistically significant.
# But a small p-value alone isn't interesting enough,
# we also want large effect sizes,
# and that's what the log(odds) and log(odds ratio) tells us.
## Eg: hd = -1.0438 + 1.2737
# The second term indicated the INCREASE in the log(odds)
# that a male has of having heart disease.
# It's the log(odds ratio) of the odds that a male will have heart disease
# over the odds that a female will have heart diseases


## (Dispersion parameter for binomial family taken to be 1)
# When we do "normal" linear regression,
# we estimate both the mean and the variance from the data.
# With logistic regression, we estimate the mean of the data,
# and the variance is derived from the mean.
# Since we are not estimating the variance from the data,
# (and, instead, just deriving it from the mean),
# it is possible that the variance is underestimated.
# If so, we can adjust the dispersion parameter in the summary() command.

## AIC: 390.12
# Akaike Information Criterion
# A measure of the relative quality of a statistical model 
# for a given set of data.
# In this context, it is just the Residual Deviance adjusted 
# for the number of parameters in the model.
# It can be used to compare different models.
# Lower AIC values indicate a better fit.

## Number of Fisher Scoring iterations: 4
# It tells us how quickly the glm() function converged on the max. likelihood 
# estimates for the coefficients.

logistic <- glm(num ~ ., data = data, family = "binomial")
summary(logistic)

## age         -0.023508   0.025122  -0.936 0.349402
# age isn't a useful predictor because it has a large p-value.
# The mddian age in our dataset is 56,
# so most of the folks are pretty old and that's why it wasn't very useful.

##     Null deviance: 409.95  on 296  degrees of freedom
## Residual deviance: 183.10  on 276  degrees of freedom
## AIC: 225.1
# The Residual Deviance and the AIC are both much smaller 
# than they were for the simple model

# Calculate McFadden's Pseudo R²
# Pull out the log-likelihood of the null model of the logistic variable
# by getting the value for the null deviance and dividing by -2
ll.null <- logistic$null.deviance / -2
# Pull out the log-likelihood of the proposed model of the logistic variable
# by getting the value for the residual deviance and dividing by -2
ll.null <- logistic$deviance / -2

# the overall effect size
(ll.null - ll.proposed) / ll.null

# use those same log-likelihoods to calculate a p-value for that R²
# using a Chi-Sq distribution
1 - pchisq(2 * (ll.proposed - ll.null), 
           df = (length(logistic$coefficients) - 1))

# create a new data.frame that contains the probabilities of having hd
# along with the actual heart disease status
predicted.data <- data.frame(
    probability.of.heart.disease = logistic$fitted.values,
    actual.heart.disease = data$num
) 

# sort the data.frame from low to high probabilities
predicted.data <- predicted.data[
    order(predicted.data$probability.of.heart.disease,
    decreasing = FALSE),]

# add a new col to the data.frame that has the rank of each sample
predicted.data$rank <- 1:nrow(predicted.data)

# install ggplot2 lib once for the first time
install.packages("ggplot2")
# load the ggplot2 lib to draw a graph (for every session)
library(ggplot2)
# so that ggplot has nice looking defaults
install.packages("cowplot")
library(cowplot)

# call ggplot() and use geom_point() to draw the data
ggplot(
    data = predicted.data,
    aes(
        x = rank,
        y = probability.of.heart.disease
    )
) +
geom_point(
    aes(color = actual.heart.disease),
    alpha = 1,
    shape = 4,
    stroke = 2
) +
xlab("Index") +
ylab("Predicted probability of heart disease")

# save the graph as a PDF
ggsave("heart_disease_probabilities.pdf")