"""
This script preprocesses, trains and evaluates a regression tree
"""

# Import modules
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Read the data
os.chdir("/Users/aleph/Desktop/MDS/semestres/2/machine learning/data/")
data = pd.read_csv("engineered_data_sample.csv")

## One-hot encode

# Create one-hot encoded 'adata'
one_hot_adat = pd.get_dummies(data['adat'])

# Drop unencoded column
data = data.drop('adat', axis = 1)

# Join the encoded df
data = data.join(one_hot_adat)

## Prepare 'X' and 'y'
X = data.loc[:, data.columns != 'hits']
y = data.loc[:, 'hits']

## Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## Train
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

## Predict and evaluate 
y_predictions = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, y_predictions)
tree_rmse = np.sqrt(tree_mse)
