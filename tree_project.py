import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions.winter_school_helper import clean_data, variable_selection_by_importance
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import seaborn as sns
sns.set()

# Read in the data
data_1 = pd.read_csv('Regression_Supervised_Test_1.csv', index_col='lotid')
data_2 = pd.read_csv('Regression_Supervised_Train.csv', index_col='lotid')

# join the data together
joined_data = pd.concat([data_1, data_2])

# Training the model
data = clean_data(joined_data)

# Split the data and log Transform the target column
X = data.drop(columns=['parcelvalue'], axis=1)
MinMaxScaler().fit_transform(X)
y = data['parcelvalue']
y_log = data['parcelvalue'].apply(np.log)

# Split the data into a train and test set. 70% train, 30% test with a random seed of 69.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.3, random_state=69)

grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=15, n_estimators=50, subsample=0.8)

# non log
grd_boost.fit(X_train, y_train)
y_pred_gb = grd_boost.predict(X_test)
print("non Log:")
print(mean_absolute_error(y_test, y_pred_gb))

print("")
# Log Transform
grd_boost.fit(X_log_train, y_log_train)
y_log_pred_gb = grd_boost.predict(X_log_test)
new_pred = np.power(np.e, y_log_pred_gb)
new_y_test = np.power(np.e, y_log_test)
print("with Log:")
print(mean_absolute_error(new_y_test, new_pred))

# Doing Tree stuff
# leaf_nodes = list(range(2, 21))
# # Decision Tree
# dc_mae_score = []
# for x in range(2, 21):
#     decisiontree = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=x)
#     decisiontree.fit(X_train, y_train)
#     y_pred_dc = decisiontree.predict(X_test)
#     dc_mae_score.append(mean_absolute_error(y_test, y_pred_dc))
#
# # Random Forest
# rf_mae_score = []
# for x in range(2, 21):
#     forest = RandomForestRegressor(min_samples_split=100, max_depth=x)
#     forest.fit(X_train, y_train)
#     y_pred_f = forest.predict(X_test)
#     rf_mae_score.append(mean_absolute_error(y_test, y_pred_f))
#
# # Gradient Boosting
# gb_mae_score = []
# for x in range(2, 21):
#     grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=x, subsample=0.8)
#     grd_boost.fit(X_train, y_train)
#     y_pred_gb = grd_boost.predict(X_test)
#     gb_mae_score.append(mean_absolute_error(y_test, y_pred_gb))
#
# # Plot results
# plt.plot(leaf_nodes, dc_mae_score, label='Decision Tree', marker='.')
# plt.plot(leaf_nodes, rf_mae_score, label='Random Forest', marker='.')
# plt.plot(leaf_nodes, gb_mae_score, label='Gradient Boosting', marker='.')
# plt.legend(loc="upper right")
# plt.xlabel('Max Depth')
# plt.ylabel('Mean Absolute Error')
# plt.title('Mean Absolute Error By Tree Type')
# plt.show()
#
# # For Gradient Boosting
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


# log transform the y data
# y_log = data['parcelvalue'].apply(np.log)
# X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=69)
#
# grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=13, n_estimators=50, subsample=0.8)
# grd_boost.fit(X_train, y_train)
# y_log_pred_gb = grd_boost.predict(X_test)
# mean_absolute_error(y_test, y_log_pred_gb)
#
# new_pred = np.power(np.e, y_log_pred_gb)
# new_y = np.power(np.e, y_test)
#
# mean_absolute_error(new_y, new_pred)
#
# np.average(y)
