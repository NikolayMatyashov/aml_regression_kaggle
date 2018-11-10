from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

import data_analyzing


def cv_score():
    m = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=30, criterion='friedman_mse'),
                          n_estimators=50, learning_rate=0.5, loss='square')
    X, y = data_analyzing.get_data()
    X = MinMaxScaler().fit_transform(X.values)
    cros_val_sores = cross_val_score(m, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


def generate_solution():
    X, y = data_analyzing.get_data()
    X = MinMaxScaler().fit_transform(X.values)
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=30, criterion='friedman_mse'),
                              n_estimators=50, learning_rate=0.5, loss='square')
    model.fit(X, y)
    X_test = data_analyzing.get_test_data()
    Y_hat = model.predict(X_test)
    S = pd.DataFrame(Y_hat, columns=['price'], index=X_test.index)
    print(S.head())
    S.to_csv('solution.csv')


if __name__ == '__main__':
    # cv_score()
    generate_solution()

