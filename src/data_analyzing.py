from datetime import datetime

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from src import feature_selection


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

    # Delete all features that has no or small effect on output
    X = X[feature_selection.selected + feature_selection.macro_selected
          + feature_selection.macro_maybe_dropped + feature_selection.maybe_dropped]

    return X


# Merge test input with macro data
macro = pd.read_csv('data/macro.csv')
Z_test = pd.read_csv('data/test.csv').merge(macro, on=['timestamp'], how='inner')
Z_test = Z_test.set_index("id").sort_index()

# Merge train input with macro data
macro2 = pd.read_csv('data/macro.csv')
Z_train = pd.read_csv('data/train.csv', index_col='id').merge(macro2, on=['timestamp'], how='inner')
train_Y = Z_train['price']
Z_train = Z_train.drop('price', axis=1)

# Prepare test and train inputs
X_prepared = prepare_x(Z_train)
X_test_prepared = prepare_x(Z_test)


# --- Getters ---
def get_normalised_data():
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X_prepared.values)
    return train_test_split(pd.DataFrame(X_scaled), train_Y, test_size=0.3)


def get_train_data():
    return train_test_split(X_prepared, train_Y, test_size=0.3)


def get_data():
    return X_prepared, train_Y


def get_test_data():
    return X_test_prepared
