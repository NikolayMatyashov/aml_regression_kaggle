import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor

from src import data_analyzing

X_train, X_test, y_train, y_test = data_analyzing.get_normalised_data()


def single_run():
    """
    Performs single run of the xgboost model (build, fit, predict and calculate mean error).
    """
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, gamma=0, subsample=0.75,
                     colsample_bytree=1, max_depth=8, n_jobs=4)
    xgb.fit(X_train, y_train)
    Y_hat = xgb.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for XGBRegressor : %.3f' % MAE)


def grid_search():
    """
    Performs grid search cross validation to find optimal parameters for xgboost model
    and prints cross validation results.
    """
    tuned_parameters = [{'n_estimators': [50, 100, 300],
                         'learning_rate': [0.05, 0.07, 0.09],
                         'gamma': [0],
                         'subsample': [0.5, 0.75, 0.9],
                         'colsample_bytree': [1],
                         'max_depth': [6, 7, 8]}]

    score = 'neg_mean_absolute_error'

    clf = GridSearchCV(XGBRegressor(), tuned_parameters, cv=5,
                       scoring=score, n_jobs=4)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    Y_hat = clf.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for XGBRegressor : %.3f' % MAE)


def cv_score():
    """
    Performs cross validation on the chosen model and prints mean error and error for each iteration.
    """
    m = XGBRegressor(n_estimators=300, learning_rate=0.05, gamma=0, subsample=0.75,
                     colsample_bytree=1, max_depth=8, n_jobs=4)
    X, y = data_analyzing.get_data()
    X = preprocessing.MinMaxScaler().fit_transform(X.values)
    cros_val_sores = cross_val_score(m, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=4)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


def plot_errors(model):

    pass


if __name__ == '__main__':
    # grid_search()
    single_run()
    # cv_score()
