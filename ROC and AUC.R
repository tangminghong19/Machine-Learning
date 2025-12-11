install.packages("pROC")
library(pROC)
library(randomForest)

set.seed(420)

num.samples <- 100

# generate 100 random values from a normal distribution
# and then sort the numbers from low to high
weight <- sort(rnorm(n=num.samples, mean=172, sd=29))

# rank the weights, scale the ranks
# compare the scaled ranks to random numbers from 0 to 1
obese <- ifelse(test=(runif(n=num.samples) < (rank(weight)/100)),
  yes=1, no=0)
obese
plot(x=weight, y=obese)

glm.fit=glm(obese ~ weight, family=binomial)
lines(weight, glm.fit$fitted.values)
# glm.fit$fitted.values contains the y-axis coordinates
# along the curve for each sample.
# It contains the estimated probabilities that each sample is obese.
roc(obese, glm.fit$fitted.values, plot=TRUE)
# to get rid of the ugly padding
par(pty = "s") # set "the plot type" to "square"
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
  xlab="False Positive Percentage", ylab="True Positive Percentage",
  col="#377eb8", lwd=4) 
# change x-axis to 1-specificity

roc.info <- roc(obese, glm.fit$fitted.values, legacy.axes=TRUE)
roc.df <- data.frame(
  tpp=roc.info$sensitivities*100,
  fpp=(1 - roc.info$specificities)*100,
  thresholds=roc.info$thresholds)
# make a data.frame that contains all of the
# True Positive Percantages and False Positive Percentages
head(roc.df)
tail(roc.df)
# the last row in roc.df corresponds to 
# the bottom left-hand corner of the ROC curve
roc.df[roc.df$tpp > 60 & roc.df$tpp < 80,]

# print AUC
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
  xlab="False Positive Percentage", ylab="True Positive Percentage",
  col="#377eb8", lwd=4, print.auc=TRUE)

# print partial AUC
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
  xlab="False Positive Percentage", ylab="True Positive Percentage",
  col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90),
  auc.polygon=TRUE, auc.polygon.col = "#377eb822")
# print.auc.x: specify where along the x-axis we want the AUC to be printed,
# otherwise the text might overlap something important.
# Set partial.auc to the range of specificity values (in terms of specificity,
# not 1 - specificity) that we wanna focus on.
# Draw partial AUC by setting auc.polygon=TRUE
# 22 makes the colour semi-transparent

rf.model <- randomForest(factor(obese) ~ weight)
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
  xlab="False Positive Percentage", ylab="True Positive Percentage",
  col="#377eb8", lwd=4, print.auc=TRUE)
plot.roc(obese, rf.model$votes[,1], percent=TRUE, col="#4daf4a", lwd=4,
  print.auc=TRUE, add=TRUE, print.auc.y=40)
# rf.model$votes[,1]: pass in the number of trees in the forest that voted correctly
# add=TRUE: this ROC curve is added to an existing graph
# print.auc.y=40: so that the AUC for the random forest is printed below
# the AUC for logistic regression

legend("bottomright", legend=c("Logistic Regression", "Random Forest"),
  col=c("#377eb8", "#4daf4a"), lwd=4)
  
par(pty = "m") # reset the graphical parameter back to its
# default value, m ("Maximum")
# Use the max amount of space provided to draw graphs