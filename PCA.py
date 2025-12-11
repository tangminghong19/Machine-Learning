import pandas as pd # pandas: panel data
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing # provides function for scaling the data before PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

genes = ['gene' + str(i) for i in range(1,101)]
wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]
data = pd.DataFrame(columns=[*wt, *ko], index=genes)
# * unpacks the "wt" and "ko" arrays so that the column names are a single array:
# [wt1, wt2, ..., ko1, ko2, ..., ko5]
# Without *, it will be an array of two arrays:
# [[wt1, ..., wt5], [ko1, ..., ko5]]

# For each gene in the index (gene1, ..., gene100),
# create 5 values for the "wt" samples nad 5 values for the "ko" samples.
for gene in data.index:
    data.loc[gene,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    data.loc[gene,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    # For each gene, select a new mean for the poisson distribution, vary between 10 and 1000.

print(data.head())
print(data.shape) # returns the dimensions of the data matrix eg. (100, 10)

# centre and scale the data
scaled_data = preprocessing.scale(data.T) # If it is columns, tranpose it before analysis
# After centering, the average value for each gene will be 0.
# After scaling, the sd for the values for each gene will be 1.

# Alternatively, more commonly used for machine learning,
scaled_data = StandardScaler().fit_transform(data.T)

# In sklearn, variation is calculated with denominator equals n.
# In R using scale() or prcomp(), the denominator is n - 1 instead.
# The latter results in larger, but unbiased, estimates of the variation.
# These differences don't affect the PCA analysis.
# The loading scores and the amount of variation per PC will be the same.
# However, these differences will have a minor effect to the final graph.
# The coordinates on the final graph come from multiplying the loading scores by the scaled values.
pca = PCA() # machine learning step
pca.fit(scaled_data) # calculate loading scores and the variation each PC accounts for
pca_data = pca.transform(scaled_data) # generate coordinates for a PCA graph based on the loading scores and the scaled data.

# calculate the percentage of variance each PC accounts for
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# put the new coordiantes (pca_data) into a matrix where the rows have sample labels and columns have PC labels
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
# add sample names to the graph
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
plt.show()
