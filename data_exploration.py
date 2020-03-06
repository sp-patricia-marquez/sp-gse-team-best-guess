## LOAD LIBRARIES ------------------------------------------------------------------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import helper_functions.winter_school_helper as hf
from sklearn.cluster import KMeans
import random
from statistics import mode,median,mean,variance

color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

sns.set()

## INITIAL TREATMENT OF THE DATA ---------------------------------------------------------------------------------------
# Read in the data --------------------------------------------------------------------
train = pd.read_csv('Data/Regression_Supervised_Train.csv', index_col='lotid')
# Drop all rows where parcelvalue is null
train = train[train['parcelvalue'].notnull()]

# Delete if there are, outliers -------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(train['totalarea'], train['parcelvalue'])
plt.ylabel('parcelvalue', fontsize=13)
plt.xlabel('totalarea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['totalarea']>15000) | (train['parcelvalue']>6000000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['totalarea'], train['parcelvalue'])
plt.ylabel('parcelvalue', fontsize=13)
plt.xlabel('totalarea', fontsize=13)
plt.show()


## TARGET VARIABLE  ----------------------------------------------------------------------------------------------------
sns.distplot(train['parcelvalue'], fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['parcelvalue'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('parcelvalue distribution')
plt.show()
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['parcelvalue'], plot=plt)
plt.show()

# Log-transformation of the target variable
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["parcelvalue_log"] = np.log1p(train["parcelvalue"])

#Check the new distribution
sns.distplot(train["parcelvalue_log"], fit=norm)
plt.show()
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train["parcelvalue_log"])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Parcel Value Log(1+x) distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train["parcelvalue_log"], plot=plt)
plt.show()

# Remove parcevalue_log outliers
sum(train['parcelvalue_log']<8) # how many?
train = train.drop(train[(train['parcelvalue_log']<8)].index)

# NULL VALUES
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='35')
sns.barplot(x=train_na.index, y=train_na)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

plt.show()

#clean_train = clean_data(train)
## DEALING WITH NAS ----------------------------------------------------------------------------------------------------
# FILL DUMMY NA COLUMNS
clean_train = hf.fill_dummy_na(train)

# REMOVE COLUMNS WITH HIGH NUMBER OF NA's AND NO TARGET VALUE AND VERY EMPTY ROWS
clean_train = hf.clean_nulls(clean_train)

# WE IMPUTE MISSING VALUES
clean_train = hf.impute_na(clean_train)


# CHECK AGAIN NULL VALUES
train_na = (clean_train.isnull().sum() / len(clean_train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head(20)

# Drop columns where can't extrapolate data
clean_train = clean_train.drop(columns=['neighborhoodcode','citycode','regioncode'], axis=1)

#Check lotarea -------------------
lot_area = clean_train['lotarea']
lot_area = lot_area[lot_area.notnull()]
hf.eval_norm_dist_after_log_trans(lot_area,'Lot Area')

lot_area_imp = clean_train['lotarea']
lot_area_imp = lot_area_imp.fillna(mode(lot_area))
hf.eval_norm_dist_after_log_trans(lot_area_imp,'Lot Area Imputed with mode')

# Impute lot area with mode:
clean_train['lotarea_mode'] = clean_train.lotarea.fillna(mode(lot_area))
clean_train['lotarea_median'] = clean_train.lotarea.fillna(median(lot_area))

# Check which transformation is better:

# Need to so something about lotarea
sns.distplot(clean_train['lotarea_mode'], fit=norm)
plt.title('Distribution of Lotarea', fontsize=15)
plt.show()

# Log transform the lotarea
clean_train["lotarea_log"] = np.log1p(clean_train["lotarea_mode"])
fig, ax = plt.subplots()
ax.scatter(clean_train['lotarea_log'], clean_train['parcelvalue_log'])
plt.ylabel('parcelvalue_log', fontsize=13)
plt.xlabel('lotarea_log', fontsize=13)
plt.show()

# After log transform
sns.distplot(clean_train['lotarea_log'], fit=norm)
plt.title('Distribution of lotarea_log After Drop', fontsize=15)
plt.show()

# WINNER: mode!!!!
clean_train['lotarea'] = clean_train.lotarea.fillna(mode(lot_area))
clean_train = clean_train.drop(columns=['lotarea_mode','lotarea_median'], axis=1)

# Check finished area -------------------
finished_area = clean_train['finishedarea']
finished_area = finished_area[finished_area.notnull()]
hf.eval_norm_dist_after_log_trans(finished_area,'Finished Area')

finished_area_imp = clean_train['finishedarea']
finished_area_imp = finished_area.fillna(mode(finished_area_imp))
hf.eval_norm_dist_after_log_trans(finished_area_imp,'Lot Area Imputed with mode')

# Impute finishedarea with mode:
clean_train['finishedarea_mode'] = clean_train.finishedarea.fillna(mode(finished_area))
clean_train['finishedarea_median'] = clean_train.finishedarea.fillna(median(finished_area))
clean_train['finishedarea_mean'] = clean_train.finishedarea.fillna(mean(finished_area))

# Check which transformation is better:

# Need to do the same with 'finishedarea'
sns.distplot(clean_train['finishedarea_mean'], fit=norm)
plt.title('Distribution of finishedarea', fontsize=15)
plt.show()

clean_train["finishedarea_log"] = np.log1p(clean_train["finishedarea_mean"])
fig, ax = plt.subplots()
ax.scatter(clean_train['finishedarea_log'], clean_train['parcelvalue_log'])
plt.ylabel('parcelvalue_log', fontsize=13)
plt.xlabel('finishedarea_log', fontsize=13)
plt.show()

sns.distplot(clean_train['finishedarea_log'], fit=norm)
plt.title('Distribution of finishedarea_log', fontsize=15)
plt.show()

# And the winner is .... NONE OF THEM!!!
# Since parcel value and finished area seem to be correlated, and also the null values are below 5%,
# we decide to drop rows with nul finished area!

clean_train = clean_train[clean_train['finishedarea'].notnull()]
clean_train = clean_train.drop(columns=['finishedarea_mean','finishedarea_median','finishedarea_mode'], axis=1)
#---------------------
summary = clean_train.describe()
summary = summary.transpose()
summary

clean_train.info()

np.sort(clean_train.columns).tolist()


## REMOVING 0-1 VARIABLES WITH LOW VARIANCE ----------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/feature_selection.html
max_val = clean_train.max()
col_names = max_val[max_val == 1].index.values

bernoulli_cols = clean_train[col_names]

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# VarianceThreshold needs array format
sel.fit_transform(bernoulli_cols.values)
# Remaining columns:
final_cols = col_names[sel.get_support(indices=True)]
# Columns to drop, since the variance is low:
drop_cols = list(filter(lambda x : x not in final_cols, col_names))
clean_train = clean_train.drop(columns=drop_cols, axis=1)


## FEATURING THE OTHER VARIABLES ----------------------------------------------------------------------------------------
# Listing the variables:
np.sort(clean_train.columns).tolist()

# -- Countycode and countycode2
hf.plot_target_vs_var(clean_train,'parcelvalue','countycode')
hf.plot_target_vs_var(clean_train,'parcelvalue','countycode2')
corrmat = clean_train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
corrmat['countycode']['countycode2']
corrmat['parcelvalue']['countycode']
corrmat['parcelvalue']['countycode2']
# Countycode and countycode2 have a high correlation (0.6) and countycode2 has slightly higher
# correlation with parcelvalue, thus we drop countycode and later will transform as dummies
clean_train = clean_train.drop(columns='countycode', axis=1)


# -- Transaction date
clean_train['transactiondate'] = pd.to_datetime(clean_train['transactiondate'])
clean_train['transactiondate'] = clean_train.transactiondate.apply(lambda x: x.strftime('%Y'))
hf.plot_target_vs_var(clean_train,'parcelvalue','transactiondate')
# Powereye, does not show significant change (pending to do a proper statistical test) --> we drop this column
clean_train = clean_train.drop(columns=['transactiondate'], axis=1)

# -- Tax year
hf.plot_target_vs_var(clean_train,'parcelvalue','taxyear')
# Powereye, does not show significant change (pending to do a proper statistical test) --> we drop this column
clean_train = clean_train.drop(columns=['taxyear'], axis=1)

# -- year
var = 'year'
hf.plot_target_vs_var(clean_train,'parcelvalue','year')
#YEAR, CHECK CLUSTERING OF YEAR TO BIN THE FEATURE TO THEN MAKE DUMMY VALUES
year_x = clean_train[['parcelvalue_log', 'year']]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(year_x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.show()

# Best out come is n_cluster = 5
# Set random seed
random.seed(69)
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(year_x)
print(kmeans.labels_)

clean_train['year_cat'] = kmeans.labels_


hf.plot_target_vs_var(clean_train,'year_cat','year')
clean_train = clean_train.drop(columns=['year'], axis=1)

# -- unitnum
print(clean_train.unitnum.value_counts())
hf.plot_target_vs_var(clean_train,'parcelvalue','unitnum')
clean_train = clean_train[clean_train['unitnum'] < 7]



## Dummy Data & Categorical data -------------------------------------------------------------------------------
dummy_columns = ['countycode2', 'year_cat']
clean_train = pd.get_dummies(clean_train, columns=dummy_columns, drop_first=True)

# Clean remaining shits from exploration ------------------------------------------------------------------------------
clean_train = clean_train.drop(columns=['finishedarea', 'lotarea','parcelvalue'], axis=1)
np.sort(clean_train.columns).tolist()

# Drop lat and long for now
clean_train = clean_train.drop(columns=['latitude', 'longitude'], axis=1)

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Getting a score for the non scaled data to compare against
decisiontree = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=15)
decisiontree.fit(X_train, y_train)
y_pred_dc = decisiontree.predict(X_test)
print("Non Scaler score: {}".format(mean_absolute_error(y_test, y_pred_dc)))

# Perform Standard Scaling on only the continuous data
scaler = StandardScaler()
# Copy X_train & X_test so we can keep the original to compare against
X_train_standardscaler = X_train.copy()
X_test_standardscaler = X_test.copy()

# Fit the scalar on the train data then transform both the train and test data
X_train_standardscaler[cont_data] = scaler.fit_transform(X_train_standardscaler[cont_data])
X_test_standardscaler[cont_data] = scaler.transform(X_test_standardscaler[cont_data])


# Check the results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(X_train['finishedarea'], ax=ax1)
sns.kdeplot(X_train['latitude'], ax=ax1)
sns.kdeplot(X_train['longitude'], ax=ax1)
sns.kdeplot(X_train['lotarea'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(X_train_standardscaler['finishedarea'], ax=ax2)
sns.kdeplot(X_train_standardscaler['latitude'], ax=ax2)
sns.kdeplot(X_train_standardscaler['longitude'], ax=ax2)
sns.kdeplot(X_train_standardscaler['lotarea'], ax=ax2)
plt.show()

# perform a Decision Tree Regression on the scaled data
decisiontree_standardscaler = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=15)
decisiontree_standardscaler.fit(X_train_standardscaler, y_train)
y_pred_dc_standardscaler = decisiontree_standardscaler.predict(X_test_standardscaler)
print("Standard Scaler score: {}".format(mean_absolute_error(y_test, y_pred_dc_standardscaler)))


# Perform MinMax Scaling on only the continuous data
scaler = MinMaxScaler()
# Copy X_train & X_test so we can keep the original to compare against
X_train_minmax = X_train.copy()
X_test_minmax = X_test.copy()

# Fit the scalar on the train data then transform both the train and test data
X_train_minmax[cont_data] = scaler.fit_transform(X_train_minmax[cont_data])
X_test_minmax[cont_data] = scaler.transform(X_test_minmax[cont_data])

# Check the results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(X_train['finishedarea'], ax=ax1)
sns.kdeplot(X_train['latitude'], ax=ax1)
sns.kdeplot(X_train['longitude'], ax=ax1)
sns.kdeplot(X_train['lotarea'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(X_train_minmax['finishedarea'], ax=ax2)
sns.kdeplot(X_train_minmax['latitude'], ax=ax2)
sns.kdeplot(X_train_minmax['longitude'], ax=ax2)
sns.kdeplot(X_train_minmax['lotarea'], ax=ax2)
plt.show()

# perform a Decision Tree Regression on the scaled data
decisiontree_minmax = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=15)
decisiontree_minmax.fit(X_train_minmax, y_train)
y_pred_dc_minmax = decisiontree_minmax.predict(X_test_minmax)
print("Min Max score: {}".format(mean_absolute_error(y_test, y_pred_dc_minmax)))


# Perform Normalizer Scaling on only the continuous data
scaler = Normalizer()
# Copy X_train & X_test so we can keep the original to compare against
X_train_normalizer = X_train.copy()
X_test_normalizer = X_test.copy()

# Fit the scalar on the train data then transform both the train and test data
X_train_normalizer[cont_data] = scaler.fit_transform(X_train_normalizer[cont_data])
X_test_normalizer[cont_data] = scaler.transform(X_test_normalizer[cont_data])

# Check the results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(X_train['finishedarea'], ax=ax1)
sns.kdeplot(X_train['latitude'], ax=ax1)
sns.kdeplot(X_train['longitude'], ax=ax1)
sns.kdeplot(X_train['lotarea'], ax=ax1)
ax2.set_title('After Normalizer Scaler')
sns.kdeplot(X_train_normalizer['finishedarea'], ax=ax2)
sns.kdeplot(X_train_normalizer['latitude'], ax=ax2)
sns.kdeplot(X_train_normalizer['longitude'], ax=ax2)
sns.kdeplot(X_train_normalizer['lotarea'], ax=ax2)
plt.show()

# perform a Decision Tree Regression on the scaled data
decisiontree_normalizer = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=15)
decisiontree_normalizer.fit(X_train_normalizer, y_train)
y_pred_dc_normalizer = decisiontree_normalizer.predict(X_test_normalizer)
print("Normalizer score: {}".format(mean_absolute_error(y_test, y_pred_dc_normalizer)))


# Checking column correlation
columns = list(clean_train.columns)
columns.remove('parcelvalue_log')
correlation_dict = {}
for col in columns:
    corr = clean_train['parcelvalue_log'].corr(clean_train[col])
    correlation_dict[col] = round(corr, 5)
    print("Column: {} has a correlation of {}".format(col, round(corr, 5)))