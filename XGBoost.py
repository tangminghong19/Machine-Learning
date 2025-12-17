import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # calculate the mean and standard deviation
import xgboost as xgb
from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/refs/heads/master/data/Telco-Customer-Churn.csv")
df.head()

df.drop(['customerID'], axis=1, inplace=True)
# axis=0: remove rows; =1: columns

# It's ok to have whitespace for XGBoost and classification (due to One-Hot Encoding), but it's not ok if we want to draw a tree.
df['MultipleLines'].replace(' ', '_', regex=True, inplace=True)
df['PaymentMethod'].replace(' ', '_', regex=True, inplace=True)
# regex: regular expression
df['MultipleLines'].unique()
df['PaymentMethod'].unique()

### Missing Data Part 1: Identifying Missing Data
# XGBoost has default behaviour for missing data. (0)
df.dtypes # look for missing data
# object: 'Yes',  'No'

df['TotalCharges'].unique()
# too many values to print (...)
# However,
# df['TotalCharges'] = pd.to_numeric(df['TotalCharges']) creates an error
# Unable to parse string " " at position ...

### Missing Data Part 2: Dealing with Missing Data, XGBoost Style
# First, identify how many rows are missing data.
# If it's a lot we might have a problem bigger than what XGBoost can deal with on its own.
# If it's not that many, we can just set them to 0.
len(df.loc[df['TotalCharges'] == ' '])
df.loc[df['TotalCharges'] == ' ']
# All 11 people with TotalCharges == ' ' have just signed up, because tenure is 0.
# Thus, their churn is 'No'.
# So, we can set TotalCharges to 0 for these 11 people.
df.loc[(df['TotalCharges'] == ' '), 'TotalCharges'] = 0
# Verify that we modified TotalCharges correctly.
df.loc[df['tenure'] == 0]

# TotalCharges still has the object datatype.
# But XGBoost only allows int, float, boolean
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.dtypes

# Replace all of the other whitespaces in all of the columns with underscores dataframe-wide.
df.replace(' ', '_', regex=True, inplace=True)

### Format Data Part 1: Split the Data into Dependent and Independent Variables
X = df.drop('Churn', axis=1).copy() # Alternatively, X = df_no_missing.iloc[:,:-1]
y = df['Churn'].copy()

### Format Data Part 2: One-Hot Encoding
# XGBoost natively supports continuous data, but not categorical data
# One-Hot Encoding works great for trees, not for linear and logistic regressions.
X_encoded = pd.get_dummies(X, columns=['gender',
                                       'Partner',
                                       'Dependents',
                                       'PhoneService',
                                       'MultipleLines',
                                       'InternetService',
                                       'OnlineSecurity',
                                       'OnlineBackup',
                                       'DeviceProtection',
                                       'TechSupport',
                                       'StreamingTV',
                                       'StreamingMovies',
                                       'Contract',
                                       'PaperlessBilling',
                                       'PaymentMethod'])
X_encoded.head()
y = y.map({'No': 0, 'Yes': 1})
y.unique()

# XGBoost uses Sparse Matrices, it only keeps track of the 1s.
# Allocate memory to the 1s.

### Build a Preliminary XGBoost Model
# The data is imbalanced.
sum(y)/len(y)
# Only 27% of the people in the dataset left the company.
# Thus when we split the data into training and testing, we will split using stratification
# in order to maintain the same percentage of people who left the company in both the training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)
sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

clf_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=None, seed=42)
# XGB uses logistic regression approach to evaluate how good the classification is.
# missing: things to represent missing data

# Create trees by passing the training data.
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_round=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])
# Build 10 more trees when the prediction doesn't improve.
# Stop when none of the 10 trees improve the prediction.
# Use AUC to evaluate how good the prediction is.
# Evaluate how many trees to build using testing dataset.

# See how the model performs on the Testing Dataset by running
# the Testing Dataset down the model and drawing a confusion matrix.
plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])
# 49% false negative. XGB model was not awesome.
# Part of the problem is that the data is imbalanced.
# People leaving costs the company a lot of money, we would like to capture more people that left.
# XGB has a parameter that helps with imbalanced data.
# Improve predictions using cross validation to optimise the parameters.
# scale_pos_weight adds a penalty to incorrectly classified minority class.
# Increase that penalty so that the tree will correctly classify it.

### Optimise Parameters using Cross Validation and GridSearch()
# XGB has a lot of hyperparameters,
# parameters that we have to manually configure and are not determined by XGB itself,
# including max_depth, learning_rate, gamma, reg_lambda.
# GridSearchCV() will test all possible combinations of the parameters for us.

# If we only care about the overall performance metric (AUC) of the prediction,
# * balance the +ve and -ve weights via scale_pos_weight
# * use AUC for evaluation

# Run GridSearchCV sequentially on subsets of parameter options, rather than all
# at once in order to optimise parameters in a shorter period of time.

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5] # XGB recommends sum(-ve instances)/sum(+ve instances)
}
# learning_rate and reg_lambda were at the ends of their range, we'll continue to try them out
param_grid = {
    'max_depth': [4],
    'learning_rate': [0.1, 0.5, 1],
    'gamma': [0.25],
    'reg_lambda': [10.0, 20, 100],
    'scale_pos_weight': [3]
}

# To speed up cross validation, and to prevent overfitting,
# use only a random subset of the data (90%) and
# use a random subset of the features (columns) (50%) per tree.
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0, # =2 if you wanna see what Grid Search is doing
    n_jobs = 10,
    cv = 3 # 3-fold                            
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)
print(optimal_params.best_params_)

### Building, Evaluating, Drawing, and Interpreting the Optimised XGBoost Model
clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic',
                            gamma=0.25,
                            learn_rate=0.1,
                            max_depth=4,
                            reg_lambda=10,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])
# People leaving costs the company a lot of money, we would like to capture more people that left.

# Build 1 tree.
# Otherwise, we'll get the average of gain and cover etc. over all trees.
clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic',
                            gamma=0.25,
                            learn_rate=0.1,
                            max_depth=4,
                            reg_lambda=10,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            n_estimators=1) # Get gain, cover etc.
clf_xgb.fit(X_train, y_train)

# weight: number of times a feature is used in a branch or root across all trees
# gain: average gain across all splits that the feature is used in
# cover: the average coverage across all splits a feature is used in
# total gain: the total gain across all splits the feature is used in
# For 1 tree, gain = total_gain, cover = total_cover
bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape': 'box',
               'style': 'filled, rounded',
               'fillcolor': '#78cbe'}
leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': '#e48038'}

xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)
# num_trees is not the number of trees to plot, but the specific tree you wanna plot.
# The default value is 0.

# Save the figure as PDF:
graph_data = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)
graph_data.view(filename='xgboost_tree_customer_churn')