library(ggplot2)
data.matrix <- matrix(nrow=100, ncol=10)
colnames(data.matrix) <- c(
  paste("wt", 1:5, sep=""),
  paste("ko", 1:5, sep="")
)
rownames(data.matrix) <- paste("gene", 1:100, sep="")
for (i in 1:100) {
  wt.values <- rpois(5, lambda=sample(x=10:1000, size=1))
  ko.values <- rpois(5, lambda=sample(x=10:1000, size=1))

  data.matrix[i,] <- c(wt.values, ko.values)
}
head(data.matrix)

pca <- prcomp(t(data.matrix), scale=TRUE, center=TRUE)
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
pca.var.per
pca.data <- data.frame(Sample=rownames(pca$x),
  X=pca$x[,1],
  Y=pca$x[,2])
pca.data

ggplot(data=pca.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() +
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep="")) +
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep="")) +
  theme_bw() +
  ggtitle("PCA graph")

### MDS and PCoA
distance.matrix <- dist(scale(t(data.matrix), center=TRUE, scale=TRUE),
  method="euclidean")
# make sure samples are rows
# center and scale the measurements for each gene (which are now columns)

# multi-dimensional scaling on the distance matrix
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)
# return eigenvalues, use them to calculate how much variation in the distance matrix
# each axis in the final MDS plot accounts for
# return the doubly (rows and cols) centered version of the distance matrix

# calculate the amount of variation each axis in the MDS plot accounts for using the eigenvalues
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)
mds.var.per

# format the data for ggplot
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
  X=mds.values[,1],
  Y=mds.values[,2])
mds.data

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using Euclidean distance")

# PCA and MDS graph look similar.
# In fact, they are exactly the same!
# This is because we used the Euclidean metric.

log2.data.matrix <- log2(data.matrix)
# average of absolute values of the log-fold change
# isn't one of the distance metrics built into dist()
# create an empty matrix
log2.distance.matrix <- matrix(0,
  nrow=ncol(log2.data.matrix),
  ncol=ncol(log2.data.matrix),
  dimnames=list(colnames(log2.data.matrix),
    colnames(log2.data.matrix)))
log2.distance.matrix
# fill the matrix
for(i in 1:ncol(log2.distance.matrix)) {
  for(j in 1:i) {
    log2.distance.matrix[i, j] <-
      mean(abs(log2.data.matrix[,i] - log2.data.matrix[,j]))
  }
}
log2.distance.matrix
# symmetrical, calculate the values for lower triangle

mds.stuff <- cmdscale(as.dist(log2.distance.matrix),
  eig=TRUE,
  x.ret=TRUE)
# convert the homemade matrix into a "true" distance matrix,
# which is the bottom triangle
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)
mds.var.per
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
  X=mds.values[,1],
  Y=mds.values[,2])
mds.data
ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using avg(logFC) as the distance")