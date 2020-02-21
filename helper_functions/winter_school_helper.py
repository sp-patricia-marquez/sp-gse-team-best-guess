import math
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def high_null_count(df, thresh):
    """
    Finds columns with a high number of null values and drops them
    :param df: Starting data frame
    :param thresh: & threshold of null values
    :return: Data frame with dropped values
    """
    cols_remove = []
    for col in df.columns:
        if df[col].isna().sum() / df.shape[0] >= thresh:
            cols_remove.append(col)

    return df.drop(columns=cols_remove, axis=1)


def remove_low_corr_columns(dataframe, target_column, thresh):
    columns = list(dataframe.columns)
    columns.remove(target_column)
    hold_cols = []
    for col in columns:
        corr = dataframe[target_column].corr(dataframe[col])
        if corr <= thresh and corr >= -thresh:
            hold_cols.append(col)

    return dataframe.drop(columns=hold_cols, axis=1)


def clean_data(data):
    data.taxdelinquencyflag = data.taxdelinquencyflag.fillna('N').replace(['Y', 'N'], [1, 0])

    # Drop all rows where parcelvalue is null
    data = data[data['parcelvalue'].notnull()]

    # Replace null is "is" columns (one that should be ether 1 or 0)
    data[['fireplace', 'tubflag']] = data[['fireplace', 'tubflag']].fillna(0)

    # Drop columns that have a very high number of null values
    data = high_null_count(data, 0.9)

    # Aircon
    # Assume all na are non aircon and change to 0/1. Drop ordinal column
    data['is_aircond'] = [0 if (x == 5 or math.isnan(x)) else 1 for x in data['aircond']]
    data = data.drop(columns='aircond', axis=1)

    # Heating
    # Assume all na are non heating and change to 0/1. Drop ordinal column
    data['is_heating'] = [0 if (x == 13 or math.isnan(x)) else 1 for x in data['heatingtype']]
    data = data.drop(columns='heatingtype', axis=1)

    # Setting numbath numfullbath, 34bath as 0
    data[['numfullbath', 'num34bath', 'numbath']] = data[['numfullbath', 'num34bath', 'numbath']].fillna(0)
    data = data.drop(columns=['num34bath', 'numfullbath'], axis=1)

    # Number of stories/pools/garage, if null then 0
    data['numstories'] = data['numstories'].fillna(1)
    data[['poolnum', 'garagenum']] = data[['poolnum', 'garagenum']].fillna(0)

    # Drop rows that have more then 75% of the data missing
    data = data.dropna(axis=0, thresh=len(data.columns)*0.75)

    # Convert country code to binary and dropping country code 2.
    # data.countycode = data.countycode.replace([6037, 6059, 6111], ['A', 'B', 'C'])
    # Dummy country column
    dummy_country = pd.get_dummies(data['countycode'], drop_first=True, prefix='countycode')
    data = data.merge(dummy_country, left_index=True, right_index=True)
    # Drop original column and country code 2 as it is very similar
    data = data.drop(columns=['countycode', 'countycode2'], axis=1)

    data = data[data['regioncode'].notnull()]
    data = data[data['citycode'].notnull()]

    # Drop columns where can't extrapolate data
    data = data.drop(columns='neighborhoodcode', axis=1)

    # if unitnum is null assume only 1 building
    data.unitnum = data.unitnum.fillna(1)

    # Set garage area to 0 if null
    data.garagearea = data.garagearea.fillna(0)

    # No way to get the year of the buiulding so drop rows with null year
    data = data[data['year'].notnull()]

    # Fill lot area and finished area with median
    data['lotarea'] = data.lotarea.fillna(stats.median(data['lotarea']))
    data['finishedarea'] = data.finishedarea.fillna(stats.median(data['finishedarea']))

    # filling quality build column
    # print(data.corr().loc['taxyear', 'qualitybuild'])
    # data.boxplot('qualitybuild', 'taxyear')
    # plt.show()

    # Fill qualitybuild based on tax year (highest correlation)
    mask_1 = data['taxyear'] == 2016.00000
    mask_2 = data['taxyear'] == 2015.00000
    data.loc[mask_1, 'qualitybuild'] = data.loc[mask_1, 'qualitybuild'].fillna(7)
    data.loc[mask_2, 'qualitybuild'] = data.loc[mask_2, 'qualitybuild'].fillna(5)

    # drop string columns
    data_clean = data.drop(columns=['transactiondate', 'tubflag', 'fireplace'], axis=1)

    return data_clean


def variable_selection_by_importance(drop_thresh, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    important_features = pd.DataFrame(model.feature_importances_ / model.feature_importances_.max(),
                                      index=X_train.columns, columns=['importance'])

    # Print feature importance sort desc
    print(important_features.sort_values('importance', ascending=False))

    y_pred_gb = model.predict(X_test)
    print("Current score with all features:")
    print(mean_absolute_error(y_test, y_pred_gb))

    # Select useless features to drop
    useless_features = important_features[important_features['importance'] < drop_thresh]
    drop_list = list(useless_features.index)
    print("Features to drop:")
    print(drop_list)

    X_train_new = X_train.drop(columns=drop_list, axis=1)
    X_test_new = X_test.drop(columns=drop_list, axis=1)

    model.fit(X_train_new, y_train)

    # New score for important features
    y_pred_gb_new = model.predict(X_test_new)
    print("New Mean Absolute Error")
    print(mean_absolute_error(y_test, y_pred_gb_new))

    return model

def scaler_grid_search(model, X, y):
    """
    Returns the mean absolute error score of each scaler
    :param model: selected model
    :param X: Feature data
    :param y: Target data
    :return: Prints the scaler and score
    """
    scalers = [Normalizer(), StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), PowerTransformer()]
    for scaler in scalers:
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=69)
        model.fit(X_train, y_train)
        y_pred_gb = model.predict(X_test)
        score = mean_absolute_error(y_test, y_pred_gb)
        print("Scaler {} has a MAE score of: {}".format(scaler, score))

def model_selecter(X, y, max_depth=30, scaler=None):
    """
    This does this
    :param X: Feature data
    :param y: Target data
    :param max_depth: Max tree depth (optional)
    :param scaler: Scaler to use on feature data (optional)
    :return:
    """
    if scaler:
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
    leaf_nodes = list(range(2, max_depth+1))

    # Decision Tree
    dc_mae_score = []
    for x in range(2, 21):
        decisiontree = DecisionTreeRegressor(min_samples_split=100, max_leaf_nodes=x)
        decisiontree.fit(X_train, y_train)
        y_pred_dc = decisiontree.predict(X_test)
        dc_mae_score.append(mean_absolute_error(y_test, y_pred_dc))

    # Random Forest
    rf_mae_score = []
    for x in range(2, 21):
        forest = RandomForestRegressor(min_samples_split=100, max_depth=x)
        forest.fit(X_train, y_train)
        y_pred_f = forest.predict(X_test)
        rf_mae_score.append(mean_absolute_error(y_test, y_pred_f))

    # Gradient Boosting
    gb_mae_score = []
    for x in range(2, 21):
        grd_boost = GradientBoostingRegressor(min_samples_split=100, max_depth=x, subsample=0.8)
        grd_boost.fit(X_train, y_train)
        y_pred_gb = grd_boost.predict(X_test)
        gb_mae_score.append(mean_absolute_error(y_test, y_pred_gb))

    # Plot results
    plt.plot(leaf_nodes, dc_mae_score, label='Decision Tree', marker='.')
    plt.plot(leaf_nodes, rf_mae_score, label='Random Forest', marker='.')
    plt.plot(leaf_nodes, gb_mae_score, label='Gradient Boosting', marker='.')
    plt.legend(loc="upper right")
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error By Tree Type')
    plt.show()