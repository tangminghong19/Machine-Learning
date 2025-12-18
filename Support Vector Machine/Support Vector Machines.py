# Getting the correct answer is a higher priority than
# understanding why we get the correct answer.
# Do not require much optimisation.
import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # data manipulation
import matplotlib.pyplot as plt # draw graphs
import matplotlib.colors as colors
from sklearn.utils import resample # downsample the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale # scale and center data
from sklearn.svm import SVC # make a svm for classification
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA # perform PCA to plot the data

df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
                   header=1) # The second line contains column names, skip the first line
df.head()

df.rename({'default payment next month' : 'DEFAULT'}, axis='columns', inplace=True)
df.drop('ID', axies=1, inplace=True)

### Missing Data Part 1: Identifying Missing Data
df.dtypes
# Good! No NA nor other character-based placeholders for missing data.

df['SEX'].unique()
df['EDUCATION'].unique()
# 0, 5, 6?
df['MARRIAGE'].unique()
# 0?
# Maybe it's missing data?

### Missing Data Part 2: Dealing with Missing Data
# scikit-learn's svm don't support datasets with missing values.
len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)])
len(df)
# Less than 1% contain missing values.
# We can remove them.
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
len(df_no_missing)
df_no_missing['EDUCATION'].unique()
df_no_missing['MARRIAGE'].unique()

### Downsample the data
# Svm are great with small datasets.
# This dataset is big enough to take a long time to optimise with cv.
# Thus, we'll downsample both categories, customers who did and did not default, to 1000 each.
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]

df_no_default_downsampled = resample(df_no_default,
                                     replace=False,
                                     n_samples=1000,
                                     random_state=42)
len(df_no_default_downsampled)
df_default_downsampled = resample(df_default,
                                  replace=False,
                                  n_samples=1000,
                                  random_state=42)
len(df_default_downsampled)
# Merge two datasets into a single dataframe.
df_downsampled = pd.concat([df_no_default_downsampled, df_default_downsampled])
len(df_downsampled)

### Format Data Part 1: Split the Data into Dependent and Independent Variables
X = df_downsampled.drop('DEFAULT', axis=1).copy()
X.head()
y = df_downsampled['DEFAULT'].copy()
y.head()

### Format Data Part 2: One-Hot Encoding
# scikit-learn SVM natively support continuous data, eg LIMIT_BAL and AGE,
# but they don't natively support categorical data.
X_encoded = pd.get_dummies(X, columns=['SEX',
                                       'EDUCATION',
                                       'MARRIAGE',
                                       'PAY_0',
                                       'PAY_2',
                                       'PAY_3',
                                       'PAY_4',
                                       'PAY_5',
                                       'PAY_6'])

### Format the Data Part 3: Centering and Scaling
# The Radial Basis Function (RBF) assumes that the data are centered and scaled.
# Each column should have mean=0 and standard deviation=1.
# We need to center and scale both the training and testing datasets separately to avoid Data Leakage.
# Data leakage occurs when info about the training dataset corrupts or influences the testing dataset.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

### Build a Preliminary Support Vector Machine
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train) # y_train is just 0 and 1, no need to be scaled
# We've built a SVM for classification.
# Let's see how it performs on the testing dataset and draw a confusion matrix.
ConfusionMatrixDisplay(clf_svm,
                       X_test_scaled,
                       y_test,
                       values_format='d',
                       display_labels=["Did not default", "Defaulted"])
# Meh.

### Optimise Parameters with Cross Validation and GridSearchCV()
# Find the best value for gamma and the regularisation parameter C.
param_grid = [
    {'C': [0.5, 1, 10, 100], # must > 0
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']}
]
# Include C = 1 and gamma = 'scale' as possible choices
# since they are the default values.
optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy', # de fault
    # scoring='balanced_accuracy', # slightly improved, C=1, gamma=0.01
    # scoring='f1', # terrible, C=0.5, gamma=1
    # scoring='f1_micro', # slighty improved, C=1, gamma=0.01
    # scoring='f1_macro', # same, C=1, gamma='scale'
    # scoring='f1_weighted', # same, C=1, gamma='scale'
    # scoring='roc_auc', terrible, C=1, gamma=0.001
    verbose=0 # =2 if you wanna see what Grid Search is doing
)

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)
# Takes time, that's why reducing the number of rows is needed.
# C=100, we'll use regularisation.

### Building, Evaluating, Drawing and Interpreting the Final SVM
clf_svm = SVC(random_state=42, C=100, gamma=0.001)
clf_svm.fit(X_train_scaled, y_train)
ConfusionMatrixDisplay(clf_svm,
                       X_test_scaled,
                       y_test,
                       values_format='d',
                       display_labels=["Did not default", 'Defaulted'])
# Just a little bit better.
# In other words, SVM was pretty good already without much optimisation.

len(df_downsampled.columns)
# 24 features, 24 dimensions.
# Use PCA to combine the 24 features into 2 orthogonal meta-features.
# Before we shrink the graph, determine how accurate the shrunken graph will be by drawing a scree plot.
pca = PCA() # By default, PCA() centers the data, but doesn't scale it.
X_train_pca = pca.fit_transform(X_train_scaled)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x', # changes apply to the x-axis
    which='both', # both major and minor ticks are affected
    bottom=False, # ticks along the bottom edge
    top=False, # ticks along the top edge
    labelbottom=False # labels along the bottom edge
)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()
# PC1 accounts for a relatively large amount of variation in the raw data, good.
# But not for PC2...

# Optimise an SVM fit to PC1 and PC2
train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]
# PC1 contains the x-axis coordinates of the data after PCA
# PC2 contains the y-axis

# center and scale the PCs
pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))
# optimise the SVM fit to the x and y-axis coordinates of the data
# after PCA dimension reduction
param_grid = [
    {'C': [1, 10, 100, 1000],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']}
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

# draw the graph
clf_svm = SVC(random_state=42, C=1000, gamma=0.001)
clf_svm.fit(pca_train_scaled, y_train)
# transform the dataset with PCA
X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

# Create a matrix of points to show the decision regions.
# The matrix will be larger than the transformed PCA points
# so that we can plot all of the PCA points on it without them being on the edge.
x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1
y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1
xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))

# Classify every point in that matrix with SVM.
# Points on one side of the classification boundary will get 0,
# and point on the other side will get 1.
Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
# Now, Z is just a long array of lots of 0s and 1s,
# which reflect how each point in the mesh was classified.
# Use reshape() so that each classification (0 or 1) corresponds to
# a speific point in the matrix.
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10))
# Use contourf() to draw a filled contour plot
# using the matrix values and classifications.
# The countours will be filled according to the
# predicted classifications (0s and 1s) in Z.
ax.contourf(xx, yy, Z, alpha=0.1)

# Create custom colours for the actual data points
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
# Draw the actual data points.
# They will be colored by their known (not predicted) classifications.
scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train,
                     cmap=cmap,
                     s=100,
                     edgecolors='k', # black
                     alpha=0.7)
# alpha=0.7: let us see if we are covering up a point

# Now create a legend
legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")
legend.get_texts()[0].set_text("No Default")
legend.get_texts()[1].set_text("Default")
# Add axis labels and titles
ax.sex_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decision surface using the PCA transformed/projected features')
plt.savefig('svm_default.png')
plt.show()
# Pink: All datapoints will be predicted to have not defaulted.
# Yellow: All datapoints will be predicted to have defaulted.
# Dots: Datapoints in the training dataset. Red: not default; Green: default
# The results are showing the training data, not the testing data.
# Do not match the graph with the confusion matrices.
# Since we only fir the SVM to the first 2 PC,
# this is only an approximation of the true classifier.