from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor

import data_analyzing
import numpy as np


def test_models(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = data_analyzing.get_train_data()

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

    # # kernel='poly', degree=1,2,5
    # # kernel='rbf', gamma=0.1, 1, 5, 10
    # m = KernelRidge(alpha=0.0001, kernel='rbf', gamma=1, degree=8)  # linear, poly, rbf
    # m.fit(X_train, y_train)
    # Y_hat = m.predict(X_test)
    # MAE = np.mean(abs(Y_hat - y_test))
    # print('MAE for Kernel ridge regression : %.3f' % MAE)

    m = DecisionTreeRegressor(max_depth=8)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Decision tree : %.3f' % MAE)

    # # criterion: mae,mse
    # m = RandomForestRegressor(max_features=0.91, n_estimators=20, bootstrap=True, criterion='mae')
    # m.fit(X_train, y_train)
    # Y_hat = m.predict(X_test)
    # MAE = np.mean(abs(Y_hat - y_test))
    # print('MAE for Random forest : %.3f' % MAE)

    # # n_estimators: 1,3,5, 100,500
    # # learning_rate: 1, 0.1, 0.01
    # # loss: ls, lad
    # m = GradientBoostingRegressor(n_estimators=300, learning_rate=1, max_depth=2, loss='lad')
    # m.fit(X_train, y_train)
    # Y_hat = m.predict(X_test)
    # MAE = np.mean(abs(Y_hat - y_test))
    # print('MAE for Gradient Boosting : %.3f' % MAE)

    # # solver: sgd, lbfgs, adam
    # # hidden_layer_sizes: [4],[40],[400],[190,120],[190,12]
    # # activation: identity, logistic, tanh, relu
    # m = MLPRegressor(hidden_layer_sizes=[190, 12], alpha=0.01, max_iter=1000,
    #                  solver='lbfgs', activation='relu')
    # m.fit(X_train, y_train)
    # Y_hat = m.predict(X_test)
    # MAE = np.mean(abs(Y_hat - y_test))
    # print('MAE for Multilayer perceptron : %.3f' % MAE)


def find_important_parameters(model, X, y):
    # print('--- Important parameters chosen by RFE ---')
    # rfe = RFE(model, 3)
    # rfe = rfe.fit(X, y)
    # # summarize the selection of the attributes
    # print(rfe.support_)
    # print(rfe.ranking_)

    print()
    print('--- Important parameters chosen by ExtraTreesClassifier ---')
    etc = ExtraTreesClassifier()
    etc.fit(X, y)
    # display the relative importance of each attribute
    print(etc.feature_importances_)


def cv(model, X, y):
    cros_val_sores = cross_val_score(model, X, y, cv=5, n_jobs=4)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


if __name__ == '__main__':
    # X, y = data_analyzing.get_data()
    # model = LogisticRegression()
    # find_important_parameters(model, X, y)

    X_train, X_test, y_train, y_test = data_analyzing.get_normalised_data()
    for i in range(0, 9):
        print('Iteration %d' % (i + 1))
        test_models(X_train.iloc[:, i:(i + 32)], X_test.iloc[:, i:(i + 32)], y_train, y_test)
        print()