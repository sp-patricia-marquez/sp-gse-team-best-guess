import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import helper_functions.winter_school_helper as hf
from sklearn.cluster import KMeans
import random

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


# Target variable ---------------------------------------------------------------------
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
# Featuring ----------------------------------------------------------------------------------------
# FILL DUMMY NA COLUMNS
clean_train = hf.fill_dummy_na(train)

# REMOVE COLUMNS WITH HIGH NUMBER OF NA's AND NO TARGET VALUE AND VERY EMPTY ROWS
clean_train = hf.clean_nulls(clean_train)

# WE IMPUTE MISSING VALUES
clean_train = hf.impute_na(clean_train)


# CHECK AGAIN NULL VALUES
train_na = (clean_train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head(20)


# Drop columns where can't extrapolate data
clean_train = clean_train.drop(columns=['neighborhoodcode','citycode','regioncode'], axis=1)

# Dummy Variables
clean_train['transactiondate'] = pd.to_datetime(clean_train['transactiondate'])
clean_train['transactiondate'] = clean_train.transactiondate.apply(lambda x: x.strftime('%Y-%m'))

# Drop more outlying data in unitnum column
print(clean_train.unitnum.value_counts())
clean_train = clean_train[clean_train['unitnum'] < 7]


var = 'year'
data = pd.concat([clean_train['parcelvalue'], clean_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='parcelvalue', data=data, showfliers= False)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.show()

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

clean_train = clean_train.drop(columns=['year'], axis=1)

# Dropping countycode2 as dupe of countycode
clean_train = clean_train.drop(columns=['countycode2'], axis=1)

# Dummy Data & Categorical data----------------------------------------------------
dummy_columns = ['countycode', 'taxyear', 'numstories', 'year_cat']
clean_train = pd.get_dummies(clean_train, columns=dummy_columns, drop_first=True)

# Data Correlation ----------------------------------------------------------------------------------------

# Need to so something about lotarea
sns.distplot(clean_train['lotarea'], fit=norm)
plt.title('Distribution of Lotarea', fontsize=15)
plt.show()

# Going to drop all row with lot area over 50,000
clean_train = clean_train.drop(clean_train[clean_train['lotarea'] > 50000].index)

sns.distplot(clean_train['lotarea'], fit=norm)
plt.title('Distribution of Lotarea After Drop', fontsize=15)
plt.show()

# Scaling and training ------------------------------------------------------------------------------------
# Need to drop 'transactiondate' as it is an object and parcelvalue as we have parcelvalue_log
clean_train = clean_train.drop(columns=['parcelvalue', 'transactiondate'], axis=1)

# Split data into feature and target (X, y)
X = clean_train.drop(columns=['parcelvalue_log'], axis=1)
y = clean_train['parcelvalue_log']

# select the categorical data
cat_data = ['tubflag',
            'poolnum',
            'fireplace',
            'numbath',
            'numbedroom',
            'qualitybuild',
            'numfireplace',
            'garagenum',
            'roomnum',
            'unitnum',
            'taxdelinquencyflag',
            'is_aircond',
            'is_heating',
            'countycode_6059.0',
            'countycode_6111.0',
            'taxyear_2016.0',
            'numstories_2.0',
            'numstories_3.0',
            'numstories_4.0',
            'year_cat_1',
            'year_cat_2',
            'year_cat_3',
            'year_cat_4']

# Select the continuous data
cont_data = [x for x in list(X.columns) if x not in cat_data]

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