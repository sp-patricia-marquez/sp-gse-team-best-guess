import pandas as pd
from sklearn.model_selection import train_test_split
from helper_functions.winter_school_helper import clean_data, remove_low_corr_columns
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import collections
import seaborn as sns
sns.set()

# Read in the data
test = pd.read_csv('Regression_Supervised_Test_1.csv', index_col='lotid')
train = pd.read_csv('Regression_Supervised_Train.csv', index_col='lotid')

# join the data together
joined_data = pd.concat([test, train])

# Training the model
data = clean_data(joined_data)

# clean_data = remove_low_corr_columns(data, 'parcelvalue', 0.1)

X = data.drop(columns=['parcelvalue'], axis=1)
y = data['parcelvalue']

# Split the data into a train and test set. 80% train, 20% test with a random seed of 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Doing Tree stuff
decisiontree = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=20)

decisiontree.fit(X_train, y_train)
important_features = pd.DataFrame(decisiontree.feature_importances_ / decisiontree.feature_importances_.max(),
                                  index=X_train.columns,
                                  columns=['importance'])
# it is common to normalize by the importance of the highest
important_features.sort_values('importance', ascending=False)

y_pred_prob_test = decisiontree.predict(X_test)
train_score = decisiontree.score(X_test, y_test)

print(collections.Counter(y_pred_prob_test))

grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=10)
grd_boost.fit(X_train, y_train)

y_pred_prob_gb = grd_boost.predict(X_test)
train_score = grd_boost.score(X_test, y_test)
print(collections.Counter(y_pred_prob_gb))

forest = RandomForestRegressor(n_estimators=50, min_samples_split=100, max_depth=10)
forest.fit(X_train, y_train)

y_pred_prob_forest = forest.predict(X_test)
train_score_forest = forest.score(X_test, y_test)