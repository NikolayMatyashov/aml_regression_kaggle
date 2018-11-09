import pandas as pd

Z_train = pd.read_csv('data/macro.csv')
Z_test = pd.read_csv('data/test.csv')

Y = Z_train['price']
X = Z_train.drop('price', axis=1)
