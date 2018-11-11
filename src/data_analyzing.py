from datetime import datetime

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from src import filtered_features


def prepare_x(X):
    """
    This method perform preprocessing of the input values.
    :param X: input values
    :return: preprocessed input values
    """
    # Transform timestamp to milliseconds
    X['timestamp'] = X['timestamp'].map(lambda t: datetime.strptime(t, "%Y-%m-%d").timestamp())

    # Use one hot encoding for string values
    X = X.join(pd.get_dummies(X['ecology'], prefix="ecology", drop_first=True))
    X = X.join(pd.get_dummies(X['product_type'], prefix="product_type", drop_first=True))
    X = X.drop(['ecology', 'product_type', 'sub_area'], axis=1)

    # Drop bugged features after merge
    X = X.drop(['child_on_acc_pre_school', 'modern_education_share',
                'old_education_build_share'], axis=1)

    # Change boolean values from 'yes'/'no' to 1/0
    boolean_parametrs = ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion',
                         'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion',
                         'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion',
                         'water_1line', 'big_road1_1line', 'railroad_1line']
    for boolean_parameter in boolean_parametrs:
        X[boolean_parameter] = X[boolean_parameter].map(lambda x: 0 if 'no' or '' else 1)

    # Deal with NaNs using mean value
    for column in X.columns:
        X[column] = X[column].fillna((X[column].mean()))

    return X


# Merge test input with macro data
macro = pd.read_csv(r'../data/macro.csv')

# Merge train input with macro data
Z_train = pd.read_csv(r'../data/train.csv', index_col='id').merge(macro, on=['timestamp'], how='inner')
Y_train = Z_train['price']

# Prepare test and train inputs
X_prepared = prepare_x(Z_train.drop('price', axis=1))


def get_train_data(normalise=False, only_important_features=True, with_test_split=False):
    X = X_prepared
    y = Y_train
    if only_important_features:
        X = X[filtered_features.selected + filtered_features.macro_selected
              + filtered_features.macro_maybe_dropped + filtered_features.maybe_dropped]
    if normalise:
        X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X.values))

    if with_test_split:
        return train_test_split(X, y, test_size=0.3)
    else:
        return X, y


def get_test_data(only_important_features=True):
    X_test = pd.read_csv(r'../data/test.csv').merge(macro, on=['timestamp'], how='inner')
    X_test = X_test.set_index("id").sort_index()
    X_test = prepare_x(X_test)

    if only_important_features:
        X_test = X_test[filtered_features.selected + filtered_features.macro_selected
                        + filtered_features.macro_maybe_dropped + filtered_features.maybe_dropped]

    return X_test
