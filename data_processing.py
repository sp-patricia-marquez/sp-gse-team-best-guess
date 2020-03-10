## LOAD LIBRARIES ------------------------------------------------------------------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import helper_functions.winter_school_helper as hf
import random



## INITIAL TREATMENT OF THE DATA ---------------------------------------------------------------------------------------
# Read in the data --------------------------------------------------------------------
train = pd.read_csv('Data/Regression_Supervised_Train.csv', index_col='lotid')
test = pd.read_csv('Data/Regression_Supervised_Test_1.csv', index_col='lotid')
# Drop all rows where parcelvalue is null


# Apply base cleaning to both data sets:
clean_train = hf.base_data_clean(train)
clean_test = hf.base_data_clean(test)

clean_train = clean_train.drop(columns=['parcelvalue'], axis=1)
clean_test = clean_test.drop(columns=['parcelvalue'], axis=1)


# Scaling and training ------------------------------------------------------------------------------------

# Split data into feature and target (X, y)
X = clean_train.drop(columns=['parcelvalue_log'], axis=1)
y = clean_train['parcelvalue_log']

# select the categorical data
cat_data = ['countycode2_2061.0',
            'countycode2_3101.0',
            'is_aircond',
            'is_heating',
            'qualitybuild',
            'year_cat_1',
            'year_cat_2',
            'year_cat_3',
            'year_cat_4']


# Select the continuous data
cont_data_log = ['lotarea_log', 'finishedarea_log']
cont_data = ['numbath', 'numbedroom', 'numfireplace', 'garagenum', 'garagearea', 'poolnum', 'roomnum', 'unitnum', 'numstories']

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Perform Normalizer Scaling on only the continuous data
scaler = Normalizer()
# Copy X_train & X_test so we can keep the original to compare against
X_train_normalizer = X_train.copy()
X_test_normalizer = X_test.copy()

# Fit the scalar on the train data then transform both the train and test data
X_train_normalizer[cont_data] = scaler.fit_transform(X_train_normalizer[cont_data])
X_test_normalizer[cont_data] = scaler.transform(X_test_normalizer[cont_data])


# Best model is a Gradient Boosting Regressor with a depth of 8
random.seed(6123)
grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=8, subsample=0.8, loss='huber')
grd_boost.fit(X_train_normalizer, y_train)
y_pred_gb = grd_boost.predict(X_test_normalizer)
# Inverse the log transform
y_test_real = np.expm1(y_test)
y_pred_gb_real = np.expm1(y_pred_gb)
print(mean_absolute_error(y_test_real, y_pred_gb_real))


# FINAL TEST OVER TEST DATASET
clean_test[cont_data] = scaler.transform(clean_test[cont_data])


X_validate = clean_test.drop(columns=['parcelvalue_log'], axis=1)
y_validate = clean_test['parcelvalue_log']

y_val_predicted = grd_boost.predict(X_validate)
y_validate_real = np.expm1(y_validate)
y_val_predicted_real = np.expm1(y_val_predicted)
print(mean_absolute_error(y_validate_real, y_val_predicted_real))

