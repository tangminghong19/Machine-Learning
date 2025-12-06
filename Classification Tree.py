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