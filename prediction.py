import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions.winter_school_helper import clean_data
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
sns.set()
# Read in the data
test = pd.read_csv('sp-crush-enemies/Data/Regression_Supervised_Test_1.csv', index_col='lotid')
train = pd.read_csv('sp-crush-enemies/Data/Regression_Supervised_Train.csv', index_col='lotid')
# join the data together
joined_data = pd.concat([test, train])
# Training the model
data = clean_data(joined_data)
# data = remove_low_corr_columns(data, 'parcelvalue', 0.1)
# Split the data and log Transform the target column
X = data.drop(columns=['parcelvalue'], axis=1)
y = data['parcelvalue'].apply(np.log)
# Split the data into a train and test set. 80% train, 20% test with a random seed of 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
leaf_nodes = list(range(2, 31))
# Decision Tree
dc_mae_score = []
for x in range(2, 31):
    decisiontree = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=x)
    decisiontree.fit(X_train, y_train)
    y_pred_dc = decisiontree.predict(X_test)
    dc_mae_score.append(mean_absolute_error(y_test, y_pred_dc))
# Random Forest
rf_mae_score = []
for x in range(2, 31):
    forest = RandomForestRegressor(n_estimators=50, min_samples_split=100, max_depth=x)
    forest.fit(X_train, y_train)
    y_pred_f = forest.predict(X_test)
    rf_mae_score.append(mean_absolute_error(y_test, y_pred_f))
# Gradient Boosting
gb_mae_score = []
for x in range(2, 31):
    grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=x, n_estimators=50, subsample=0.8)
    grd_boost.fit(X_train, y_train)
    y_pred_gb = grd_boost.predict(X_test)
    gb_mae_score.append(mean_absolute_error(y_test, y_pred_gb))
plt.plot(leaf_nodes, dc_mae_score, label='Decision Tree')
plt.plot(leaf_nodes, rf_mae_score, label='Random Forest')
plt.plot(leaf_nodes, gb_mae_score, label='Gradient Boosting')
plt.legend(loc="upper right")
plt.xlabel('Max Depth')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error By Tree Type')
plt.show()



# Stability Check of our model
test_accuracy_argmax = []  # the maximal test accuracy achieved for each split
importance_char = []  # the variable char_! importance
#
for bootsam in np.arange(100):
 # split randomly dataset; do not fix the seed to see variation
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 #
 # First search for depth
 test_accuracy = []
 complexity_value = []
 for max_leaf_nodes in np.arange(8, 12):
     grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=max_leaf_nodes, n_estimators=50, subsample=0.8)
     grd_boost.fit(X_train, y_train)
     y_pred = grd_boost.predict(X_test)
     test_accuracy.append(mean_absolute_error(y_test, y_pred))
     complexity_value.append(max_leaf_nodes)
 test_accuracy_argmax.append(complexity_value[np.argmin(test_accuracy)])
 #
 # print(f"Optimum max leaf {complexity_value[np.argmax(test_accuracy)]}")
 # Then find and store the relative importance of fare for the chosen tree

 leaf_node_chosen=complexity_value[np.argmin(test_accuracy)]
 grd_boost =  GradientBoostingRegressor(min_samples_split=100, max_depth=leaf_node_chosen, n_estimators=50, subsample=0.8)
 grd_boost.fit(X_train, y_train)
 important_features = pd.DataFrame(grd_boost.feature_importances_ / grd_boost.feature_importances_.max(), index=X.columns,
                                   columns=['importance'])

# Print the results in a convenient manner
result = pd.DataFrame(test_accuracy_argmax, columns=["depth"])
result["score_char"] = importance_char
result.plot(x="depth", y="score_char", kind="scatter")

result.describe()
