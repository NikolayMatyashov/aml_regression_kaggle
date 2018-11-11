import numpy
from datetime import datetime

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

import feature_selection


def prepare_x(X):
    # Drop or encode string columns
    # transform timestamp to milliseconds
    X['timestamp'] = X['timestamp'].map(lambda t: datetime.strptime(t, "%Y-%m-%d").timestamp())

    X = X.join(pd.get_dummies(X['ecology'], prefix="ecology", drop_first=True))
    X = X.join(pd.get_dummies(X['product_type'], prefix="product_type", drop_first=True))
    X = X.drop(['ecology', 'product_type', 'sub_area'], axis=1)

    # Drop bugged features after merge
    X = X.drop(['child_on_acc_pre_school', 'modern_education_share',
                'old_education_build_share'], axis=1)

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

    return X


macro = pd.read_csv('data/macro.csv')
Z_test = pd.read_csv('data/test.csv').merge(macro, on=['timestamp'], how='inner')
Z_test = Z_test.set_index("id").sort_index()

macro2 = pd.read_csv('data/macro.csv')
Z_train = pd.read_csv('data/train.csv', index_col='id').merge(macro2, on=['timestamp'], how='inner')
train_Y = Z_train['price']
Z_train = Z_train.drop('price', axis=1)

X_prepared = prepare_x(Z_train)
X_test_prepared = prepare_x(Z_test)

def get_normalised_data():
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X_prepared.values)
    return train_test_split(pd.DataFrame(X_scaled), train_Y, test_size=0.3)


def get_train_data():
    return train_test_split(X_prepared, train_Y, test_size=0.3)


def get_data():
    return X_prepared, train_Y


def get_test_data():
    return X_test_prepared


# def cv_score():
#     m = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=30, criterion='friedman_mse'),
#                           n_estimators=50, learning_rate=0.5, loss='square')
#     X, y = data_analyzing.get_data()
#     X = MinMaxScaler().fit_transform(X.values)
#     cros_val_sores = cross_val_score(m, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
#     print("Average score: %.3f" % np.mean(cros_val_sores))
#     print(cros_val_sores)
#

def generate_solution():

    scaler = MinMaxScaler()
    scaler.fit(X_prepared.values)
    X_train = scaler.transform(X_prepared.values)
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=30, criterion='friedman_mse'),
                              n_estimators=50, learning_rate=0.5, loss='square')

    model.fit(X_train, train_Y)
    index = X_test_prepared.index
    X_test = scaler.transform(X_test_prepared.values)
    Y_hat = model.predict(X_test)

    S = pd.DataFrame(Y_hat, columns=['price'], index=index)
    print(S.head())
    S.to_csv('solution.csv')


if __name__ == '__main__':
    # cv_score()
    generate_solution()
