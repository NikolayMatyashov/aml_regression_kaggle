import numpy as np
from catboost import CatBoostRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score

from src import data_analyzing

X_train, X_test, y_train, y_test = data_analyzing.get_train_data(normalise=True)


def single_run():
    """
    Performs single run of the catboost model (build, fit, predict and calculate mean error).
    """
    catboost = CatBoostRegressor(learning_rate=0.25)
    catboost.fit(X_train, y_train)
    Y_hat = catboost.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for CatBoostRegressor : %.3f' % MAE)


def grid_search():
    """
    Performs grid search cross validation to find optimal parameters for xgboost model
    and prints cross validation results.
    """
    tuned_parameters = [{'learning_rate': [0.25, 0.5, 0.75],
                         'max_depth': [5, 6, 7, 8]}]

    score = 'neg_mean_absolute_error'

    clf = GridSearchCV(CatBoostRegressor(), tuned_parameters, cv=5,
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
    y_true, y_pred = y_test, clf.predict(X_test)
    MAE = np.mean(abs(y_pred - y_true))
    print('MAE for CatBoostRegressor : %.3f' % MAE)
    print()


def cv_score():
    """
    Performs cross validation on the chosen model and prints mean error and error for each iteration.
    """
    m = CatBoostRegressor(learning_rate=0.25)
    X, y = data_analyzing.get_train_data()
    X = preprocessing.MinMaxScaler().fit_transform(X.values)
    cros_val_sores = cross_val_score(m, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=4)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


if __name__ == '__main__':
    # grid_search
    # cv()
    single_run()
