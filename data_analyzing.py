from datetime import datetime

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import feature_selection

# Z_test = pd.read_csv('data/test.csv', index_col='id')
Z_train = pd.read_csv('data/train.csv', index_col='id')
macro = pd.read_csv('data/macro.csv')

Z_train = Z_train.merge(macro, on=['timestamp'], how='inner')

Y = Z_train['price']
X = Z_train.drop('price', axis=1)

# Drop or encode string columns
# transform timestamp to milliseconds
X['timestamp'] = X['timestamp'].map(lambda t: datetime.strptime(t, "%Y-%m-%d").timestamp())

X = X.join(pd.get_dummies(X['ecology'], prefix="ecology", drop_first=True))
X = X.join(pd.get_dummies(X['product_type'], prefix="product_type", drop_first=True))
X = X.drop(['ecology', 'product_type', 'sub_area'], axis=1)

# Drop bugged features after merge
X = X.drop(['child_on_acc_pre_school', 'modern_education_share',
            'old_education_build_share'], axis=1)

# # Take parameters without neighbours and macro
# X = X.loc[:, ['full_sq', 'life_sq', 'floor', 'max_floor', 'material',
#               'build_year', 'num_room', 'kitch_sq', 'state', 'area_m']]

# Boolean values
boolean_parametrs = ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion',
                     'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion',
                     'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion',
                     'water_1line', 'big_road1_1line', 'railroad_1line']
for boolean_parametr in boolean_parametrs:
    X[boolean_parametr] = X[boolean_parametr].map(lambda x: 0 if 'no' or '' else 1)

# Deal with NaNs
for column in X.columns:
    X[column] = X[column].fillna((X[column].mean()))

X = X[feature_selection.selected + feature_selection.macro_selected
      + feature_selection.macro_maybe_dropped + feature_selection.maybe_dropped]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


def get_normalised_data():
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
    return train_test_split(pd.DataFrame(X_scaled), Y, test_size=0.3)


def get_train_data():
    return X_train, X_test, y_train, y_test


def get_data():
    return X, Y
