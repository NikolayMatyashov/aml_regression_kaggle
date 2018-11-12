import numpy as np
import xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from src import data_analyzing


def test_models(X_train, X_test, y_train, y_test):
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=7)
    xgb.fit(X_train, y_train)
    Y_hat = xgb.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for XGBRegressor : %.3f' % MAE)

    # alpha: 0.001, 0.5, 50, 1000
    m = Ridge(alpha=0.001)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Ridge : %.3f' % MAE)

    # alpha: 0.001, 0.5, 50, 1000
    m = Lasso(alpha=0.5)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Lasso : %.3f' % MAE)

    m = KernelRidge(alpha=0.0001, kernel='rbf', gamma=1, degree=8)  # linear, poly, rbf
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Kernel ridge regression : %.3f' % MAE)

    m = DecisionTreeRegressor(max_depth=8)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Decision tree : %.3f' % MAE)

    # criterion: mae,mse
    m = RandomForestRegressor(max_features=0.91, n_estimators=20, bootstrap=True, criterion='mae')
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Random forest : %.3f' % MAE)

    m = GradientBoostingRegressor(n_estimators=300, learning_rate=1, max_depth=2, loss='lad')
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Gradient Boosting : %.3f' % MAE)

    m = MLPRegressor(hidden_layer_sizes=[190, 12], alpha=0.01, max_iter=1000,
                     solver='lbfgs', activation='relu')
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Multilayer perceptron : %.3f' % MAE)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_analyzing.get_train_data(normalise=True, with_test_split=True)
    test_models(X_train, X_test, y_train, y_test)
