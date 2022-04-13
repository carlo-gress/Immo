"""
This script preprocesses, trains and evaluates a decision tree regressor and a random forest regressor
"""

# Import modules
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error

# Read the data
os.chdir("/Users/aleph/Desktop/MDS/semestres/2/machine learning/data/")
data = pd.read_csv("curated_data.csv")

## One-hot encode 'adata'

# Create one-hot encoded 'adat'
one_hot_adat = pd.get_dummies(data['adat'])

# Drop unencoded column
data = data.drop('adat', axis = 1)

# Join the encoded df
data = data.join(one_hot_adat)

## Prepare the predictor vector 'X' and labels 'y', make sure to leave out the 'obid' variable out
X = data.drop(['obid', 'hits'], axis = 1)
y = data['hits']

## Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## Specify and run models

# Linear regression
reg = LinearRegression()
start = time.process_time()
reg.fit(X_train, y_train)
end = time.process_time()
reg_time = end - start
reg_predictions = reg.predict(X_test)
reg_mse = mean_squared_error(y_test, reg_predictions)
reg_rmse = np.sqrt(reg_mse)
reg_rmse

# Poisson regression
poisson = PoissonRegressor()
start = time.process_time()
poisson.fit(X_train, y_train)
end = time.process_time()
poisson_time = end - start
poisson_predictions = poisson.predict(X_test)
poisson_mse = mean_squared_error(y_test, poisson_predictions)
poisson_rmse = np.sqrt(poisson_mse)
poisson_rmse

# Regression tree regressor
tree_reg = DecisionTreeRegressor()
start = time.process_time()
tree_reg.fit(X_train, y_train)
end = time.process_time()
tree_reg_time = end - start
tree_predictions = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# Random forest regressor
forest_reg = RandomForestRegressor()
start = time.process_time()
forest_reg.fit(X_train, y_train)
end = time.process_time()
forest_reg_time = end - start
forest_predictions = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# Multi-layer perceptron regressor
start = time.process_time()
mlp = MLPRegressor(hidden_layer_sizes =(64, 64, 64), activation= "relu", random_state = 1, max_iter = 2000)
mlp.fit(X_train, y_train)
end = time.process_time()
mlp_time = end - start
mlp_predictions = mlp.predict(X_test)
mlp_mse = mean_squared_error(y_test, mlp_predictions)
mlp_rmse = np.sqrt(mlp_mse)
mlp_rmse
