import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns


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
train = pd.read_csv('sp-crush-enemies/Data/Regression_Supervised_Train.csv', index_col='lotid')
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

# Remove ordered values

# Featuring ----------------------------------------------------------------------------------------
# Remove empty columns


data_f = train.drop(columns=['parcelvalue','parcelvalue_log'], axis=1)

all_data_na = (data_f.isnull().sum() / len(data_f)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# Data Correlation ----------------------------------------------------------------------------------------