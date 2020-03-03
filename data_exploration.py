import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import helper_functions.winter_school_helper as hf
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
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


# Scaling and training ------------------------------------------------------------------------------------
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
cont_data = [x for x in list(clean_train.columns) if x not in cat_data]

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import Normalizer, StandardScaler
# column_trans = ColumnTransformer([('scaler', StandardScaler(),2], remainder='passthrough')
# column_trans.fit_transform(X)