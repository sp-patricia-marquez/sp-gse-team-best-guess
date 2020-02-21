import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions.winter_school_helper import clean_data, scaler_grid_search, model_selecter
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import seaborn as sns
sns.set()

np.random.seed(31415)

# Read in the data
data_1 = pd.read_csv('Data/Regression_Supervised_Test_1.csv', index_col='lotid')
data_2 = pd.read_csv('Data/Regression_Supervised_Train.csv', index_col='lotid')

# join the data together
joined_data = pd.concat([data_1, data_2])

# Training the model
data = clean_data(joined_data)

# Split the data and log Transform the target column
X = data.drop(columns=['parcelvalue'], axis=1)
y = data['parcelvalue']

# Split the data into a train and test set. 70% train, 30% test with a random seed of 69.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)


grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=15, n_estimators=50, subsample=0.8)

scaler_grid_search(grd_boost, X, y)

# non log
# grd_boost.fit(X_train, y_train)
# y_pred_gb = grd_boost.predict(X_test)
# print("non Log:")
# print(mean_absolute_error(y_test, y_pred_gb))


# Doing Tree stuff
model_selecter(X, y, max_depth=30)

# For Gradient Boosting
# print("Min Mean Absolute Error: {}".format(round(min(gb_mae_score), 4)))
# print("")
# for x, y in zip(leaf_nodes, gb_mae_score):
#     print("Max depth: {} | MAE score: {}".format(x, round(y, 4)))

# Checking results for removing useless features
# Select the best model and depth
# grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=15, n_estimators=50, subsample=0.8)
# Run function to remove useless features
# variable_selection_by_importance(0.005, grd_boost, X_train, X_test, y_train, y_test)


# Stability Check of our model using cross validation
# scores = cross_val_score(grd_boost, X, y, cv=5)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
