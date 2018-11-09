import pandas as pd

from sklearn.model_selection import train_test_split

Z_train = pd.read_csv('data/train.csv')
Z_test = pd.read_csv('data/test.csv')

Y = Z_train['price']
X = Z_train.drop('price', axis=1)

# Drop string columns
X = X.drop(['ecology', 'sub_area', 'product_type', 'timestamp'], axis=1)

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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


def get_train_data():
    return X_train, X_test, y_train, y_test


def get_data():
    return X, Y