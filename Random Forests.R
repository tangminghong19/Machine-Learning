library(ggplot2)
library(cowplot)
install.packages("randomForest")
library(randomForest)
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
# read the dataset into R from the url
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
    "hd"
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
data$hd <- ifelse(test = data$hd == 0, yes = "Healthy", no = "Unhealthy")
data$hd <- as.factor(data$hd)
str(data)

# set the seed for the random number generator
set.seed(42)
data.imputed <- rfImpute(hd ~ ., data = data, iter = 6) 
# iter is where we specify how many random forests rfImpute() should build
# After each iteration, rfImpute() prints out the Out-of-Bag(OOB) error rate
# Now, build a proper random forest
model <- randomForest(hd ~ ., data = data.imputed, proximity = TRUE)
# proximity = TRUE indicates that we want randomForest() 
# to return the proximity matrix, in order to cluster the samples
model
## Type of random forest: classification
# regression if we wanna predict weight or height
# unsupervised if we completely ommited the thing the random forest 
# supposed to predict
## No. of variables tried at each split: 3
# Classification tree: square root of the number of vars
# Regression tree: number of variables divided by 3
## OOB estimate of  error rate: 16.83%
# 83.17% of the OOB samples were correctly classified

# To see if 500 trees are enough for optimal classification, plot the error rates:
oob.error.data <- data.frame(
  Trees = rep(1:nrow(model$err.rate), times = 3),
  Type = rep(c("OOB", "Healthy", "Unhealthy"), each = nrow(model$err.rate)),
  Error = c(model$err.rate[,"OOB"],
    model$err.rate[,"Healthy"],
    model$err.rate[,"Unhealthy"]))
# err.rate: a matrix within model
ggplot(data = oob.error.data, aes(x = Trees, y = Error)) +
  geom_line(aes(color = Type))
# In general, we see the error rates decrease when our random forest has more trees
# If we added more trees, would the error rate go down further?
model <- randomForest(hd ~ ., data = data.imputed, ntree = 1000, proximity = TRUE)
model
# By OOB estimate of error rate and confusion matrix, 
# we didn't do a better job classifying patients.
oob.error.data <- data.frame(
  Trees = rep(1:nrow(model$err.rate), times = 3),
  Type = rep(c("OOB", "Healthy", "Unhealthy"), each = nrow(model$err.rate)),
  Error = c(model$err.rate[,"OOB"],
    model$err.rate[,"Healthy"],
    model$err.rate[,"Unhealthy"]))
ggplot(data = oob.error.data, aes(x = Trees, y = Error)) +
  geom_line(aes(color = Type))
# The error rates stabilise after 500 trees.

# Now, we need to make sure that we are considering the optimal number of vars
# at each internal node in the tree.
oob.values <- vector(length = 10)
for (i in 1:10) {
  temp.model <- randomForest(hd ~ ., data = data.imputed, mtry = i, ntree = 1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate), 1] 
  # last row, first col, ie the OOB error rate when all 1000 trees have been made
}
oob.values

# Lastly, use the random forest to draw an MDS plot with samples 
# to show how they related to each other.
distance.matrix <- dist(1 - model$proximity)
mds.stuff <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)
# calculate the percentage of variation in the distance matrix that the X and Y axes account for
mds.var.per <- round(mds.stuff$eig / sum(mds.stuff$eig) * 100, 1)
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample = rownames(mds.values),
  X = mds.values[,1],
  Y = mds.values[,2],
  Status = data.imputed$hd)
ggplot(data = mds.data, aes(x = X, y = Y, label = Sample)) +
  geom_text(aes(color = Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep = "")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep = "")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")