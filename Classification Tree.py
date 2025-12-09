import pandas as pd # to load and manipulate data and for One-Hot Encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.model_selection import cross_val_score # to perform cross-validation
from sklearn.metrics import confusion_matrix # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix # to plot a confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

# df stands for DataFrame
# When pd reads in data, it returns a dataframe, which is like a spreadsheet.
# The data are organised in rows and columns.
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', 
                 header=None)

# print the first 5 rows of the dataframe
df.head()

df.columns = ['age',
              'sex',
              'cp',
              'restbp',
              'chol',
              'fbs',
              'restecg',
              'thalach',
              'exang',
              'oldpeak',
              'slope',
              'ca',
              'thal',
              'hd']
df.head()

# dtypes tells us the data type of each column
df.dtypes
## ca          object
## thal        object
# object datatypes are used when there are mixture of things

#print out unique values in 'ca' column
df['ca'].unique()
df['thal'].unique()

# sklearn's classification trees do not support dataset with missing values
# so, we can either delete those missing data from the training set, or impute values for the missing data

# len() prints out the number of rows
# loc[], short for "location", allows us to specify which rows we want
# so we say we want any row with '?' in the 'ca' column
# OR
# any row with '?' in the 'thal' column
len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

# print out the rows with missing data
df.loc[(df['ca'] == '?') | (df['thal'] == '?')]

len(df)

# 6 of the 303 rows, or 2%, contain missing values.
# Since 303 - 6 = 297 is still plenty of data to train a classification tree,
# we will simply delete those rows with missing data,
# rather than trying to impute values for the missing data.

# select all rows that do not contain missing values
# and save them in a new dataframe called "df_no_missing"

df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]

len(df_no_missing)
df_no_missing['ca'].unique()
df_no_missing['thal'].unique()

# ca and thal columns are still object datatypes

### Split the data into dependent and independent variables
# use X to represent the columns of data that we'll use to make classifications
# use y to represent the column of data that we want to predict
# deal with missing data before splitting into X and y to ensure that each row in X correctly corresponds 
# with the appropriate value in y

# by default, pandas uses copy by REFERENCE
# using copy() to copy the data by VALUE
# to ensure that the original data df_no_missing is not modified when we modify X and y

X = df_no_missing.drop('hd', axis=1).copy() # alternatively: X = df_no_missing.iloc[:, :-1]
X.head()

y = df_no_missing['hd'].copy() # alternatively: y = df_no_missing.iloc[:, -1]

# continue formatting X so that it's suitable for making a Decision Tree

### One-Hot Encoding
X.dtypes

# scikit-learn decision trees natively support continuous data, but not categorical data (eg. cp)
# thus we have to convert a column of categorical data into multiple columns of binary values.
# One-Hot Encoding!

# If we treat the four options for cp as continuous data,
# we'd assume that 4 is more similar to 3 than it is to 1 or 2.
# That menas the decision tree would be more likely to cluster the patients with 4s and 3s together
# than the patients with 4s and 1s together.
# If we treat the four options for cp as categorical data,
# then we treat each one as a separate category that is no more or less similar to any of the other categories.
# Thus, the likelihood of the decision tree clustering patients with 4s and 3s together
# is the same as the likelihood of the decision tree clustering patients with 4s and 1s together.
# This approach makes more sense for categorical data.

X['cp'].unique()

## Ways to do One-Hot Encoding in Python:
## 1. scikit-learn.ColumnTransformer()
# It creates a persistent function that can validate data that we get in the future.
# Eg. If we build a decision tree using a categorical variable favourite colour that has red, blue, and green options,
# then ColumnTransformer() can remember those options and later on when the decision tree is used in a production system,
# if someone says their favourite colour is yellow,
# ColumnTransformer() can throw an error or handle it in some other way.
# However, it turns the data into an array and looses all of the column names,
# making it harder to verify that the usage of ColumnTransformer() worked as we intended it to.
## 2. pandas.get_dummies()
# It leaves the data in a dataframe and retains the column names,
# making it much easier to verify that it worked as intended.
# However, it does not have the persistent behaviour of ColumnTransformer().

pd.get_dummies(X, columns = ['cp']).head()
# get_dummies() puts all of the columns it doesn't modify to the left,
# and the new one-hot encoded columns to the right.

# In a real situation, we should verify all 5 of these columns only contain the accepted categories.

X_encoded = pd.get_dummies(X, columns = ['cp',
                                         'restecg',
                                         'slope',
                                         'thal'])
X_encoded.head()

df['sex'].unique()
df['fbs'].unique()
df['exang'].unique()
# Since sex, fbs, and exang only have 2 categories (0 and 1),
# we do not need to One-Hot Encode them.
# So, we're done formatting the data for the Classification Tree!

# y doesn't just contain 0s and 1s
y.unique()

y_not_zero_index = y > 0
y[y_not_zero_index] = 1 # set each non-zero value in y to 1
y.unique() # verify that y only contains 0 and 1 now

### Build A Preliminary Classification Tree (not optimised)
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded,
                                                    y,
                                                    random_state = 42) # default 70 30
# create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# plot the classification tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt,
          filled = True,
          rounded = True,
          class_names = ['No HD', 'HD'],
          feature_names = X_encoded.columns)

# Let's see how it performs on the Testing Dataset 
# by running the Testing Dataset down the tree
# and drawing a Confusion Matrix
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels = ['Does not have HD', 'Has HD'])

# CAN WE DO BETTER?
# The classification tree looks overfit to the training data.
# Let's prune the tree.

### Cost Complexity Pruning Part 1: Visualise alpha
# parameters like max_depth, min_samples are designed to reduce overfitting
# but cost complexity pruning can simplify the whole process of finding a smaller tree

# One way to find the optimal value for alpha is to plot the accuracy of the tree as a function of different values.
# Do this for both training dataset and testing dataset.

# Extract the different values of alpha that are available for this tree
# and build a pruned tree for each value for alpha.
# We omit the max value for alpha with ccp_alphas = ccp_alphas[:-1] because it would prune all leaves,
# leaving us with only a root instead of a tree.

path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
# ccp: cost complexity pruning
ccp_alphas = path.ccp_alphas # extract different values for alpha
ccp_alphas = ccp_alphas[:-1] # exclude the max value for alpha

clf_dts = [] # create an array that we'll put decision trees into

# create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

# Graph the accuracy of the trees using the training dataset and the testing dataset as a function of alpha
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing datasets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# The testing dataset hits its max value when alpha is about 0.016
# When we apply ccp to a classification tree, values for alpha go from 0 to 1, because GINI scores go from 0 to 1.
# Values for alpha for a regression tree can be much larger since the sum of squared residuals can go from 0 to positive infinity.

# How do we know we used the best training dataset and testing dataset?
# 10-Fold Cross Validation!

### Cost Complexity Pruning Part 2: Cross Validation for Finding the Best Alpha
# a better way to determine alpha
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016) # create the tree with ccp_alpha=0.016

# use 5-fold cross validation to create 5 different training and testing datasets that are then used to train and test the tree
# use 5-fold as we don't have tons of data
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

# The plotted graph shows that using different training and testing data with the same alpha resulted in different accuracies,
# suggesting that alpha is sensitive to the datasets.
# Thus, use cross validation to find the optimal value for ccp_alpha

# create an array to store the results of each fold during cross validation
alpha_loop_values = []

# For each candidate value for alpha, we will run 5-fold cross validation.
# We will then store the mean and standard deviation of the scores (the accuracy) for each call
# to cross_val_score in alpha_loop_values
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append({ccp_alpha, np.mean(scores), np.std(scores)})

# Draw a graph of the means and standard deviations of the scores
# for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')

# We need to set ccp_alpha to something closet to 0.014.
# Find the exact value with:
alpha_results[(alpha_results['alpha'] > 0.014)
              &
              (alpha_results['alpha'] < 0.015)]

# Store the ideal value for alpha
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
                                &
                                (alpha_results['alpha'] < 0.015)]['alpha']
ideal_ccp_alpha

# Python thinks that ideal_ccp_alpha is a series, which is a type of array.
# We can tell because when we print it out, we get two bits of stuff.
# The first one is 20 (index in the series), the second is 0.014225.
# Convert it from a series to a float:
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha

### Building, Evaluating, Drawing, and Interpreting the Final Classification Tree
clf_dt_pruned = DecisionTreeClassifier(random_state=42,
                                       ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

# Draw a confusion matrix to see if the pruned tree does better:
cm = confusion_matrix(y_test, clf_dt_pruned.predict(X_test), labels = clf_dt_pruned.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Does not have HD', "Has HD"])
disp.plot()
plt.show()

# Draw the pruned tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", "HD"],
          feature_names=X_encoded.columns)

# In each node, we have the variable (column name) and the threshold for splitting the observations.
# The darker the colour, the lower the GINI impurity, the better the split.
# Leaves have no column names.
# Which variables are associated to greater drop of GINI? They are the most influential.