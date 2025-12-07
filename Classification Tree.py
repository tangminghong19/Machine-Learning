import pandas as pd # to load and manipulate data and for One-Hot Encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.model_selection import cross_val_score # to perform cross-validation
from sklearn.metrics import confusion_matrix # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix # to plot a confusion matrix

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