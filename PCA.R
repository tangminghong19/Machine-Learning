data.matrix <- matrix(nrow=100, ncol=10)
colnames(data.matrix) <- c(
  paste("wt", 1:5, sep=""), # wild type: normal, every day samples
  paste("ko", 1:5, sep="") # knock-out: samples missing a gene because we knocked it out
)
rownames(data.matrix) <- paste("gene", 1:100, sep="")
for (i in 1:100) {
  wt.values <- rpois(5, lambda=sample(x=10:1000, size=1)) # Poisson distribution
  ko.values <- rpois(5, lambda=sample(x=10:1000, size=1))

  data.matrix[i,] <- c(wt.values, ko.values)
}
head(data.matrix)

pca <- prcomp(t(data.matrix), scale=TRUE)
# By default, prcomp() expects the samples to be rows and the genes to be columns.
# Thus, transpose the data.matrix.
# If not, we will get a graph that showss how the genes are related to each other.
# prcomp() returns:
# 1. x: the PCs for drawing a graph
# 2. sdev
# 3. rotation

# Use the first two columns in x to draw a 2-D plot that uses the first two PCs.
plot(pca$x[,1], pca$x[,2]) # PC1, PC2
# The first PC accounts for the most variation in the original data

# calculate how much variation in the original data each PC accounts for
pca.var <- pca$sdev^2
# the percentage of variation that each PC accounts for
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
barplot(pca.var.per, main="Scree Plot", xlab="Principal Component",
  ylab="Percent Variation")
# There's a big difference between the two clusters in plot(pca$x[,1], pca$x[,2])

library(ggplot2)

pca.data <- data.frame(Sample=rownames(pca$x),
  X=pca$x[,1],
  Y=pca$x[,2])
pca.data

ggplot(data=pca.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() + # plot the labels, rather than dots or other shapes
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep="")) +
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep="")) +
  theme_bw() + # makes the graph's background white
  ggtitle("My PCA Graph")

# Look at the loading scores for PC1 only
# since it accounts for 91.5% of the variation in the data
loading_scores <- pca$rotation[,1]
gene_scores <- abs(loading_scores)
# Genes that push samples to the left side of the graph have large -ve values,
# genes that push samples to the right have large +ve values.
# Use abs() to sort based on the numbers' magnitude rather than from high to low
gene_score_ranked <- sort(gene_scores, decreasing=TRUE)
top_10_genes <- names(gene_score_ranked[1:10])
top_10_genes

pca$rotation[top_10_genes, 1] # show the scores and signs