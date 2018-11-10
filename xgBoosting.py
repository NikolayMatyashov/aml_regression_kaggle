from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

from xgboost import XGBRegressor
import data_analyzing
import numpy as np

N = 7

X_train, X_test, y_train, y_test = data_analyzing.get_normalised_data()


def run():
    m = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                     colsample_bytree=1, max_depth=8)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for XGBRegressor : %.3f' % MAE)


def cv():
    tuned_parameters = [{
        'n_estimators': [100],
        'criterion': ["mse"],
        'max_depth': [20],
        'bootstrap': [True],
        'random_state': [0],
    }]

    score = 'neg_mean_absolute_error'

    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5,
                       scoring=score, n_jobs=-1)

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
    print('MAE: %.3f' % MAE)
    print()


def cv_score():
    m = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=20, bootstrap=True, random_state=0)
    X, y = data_analyzing.get_data()
    cros_val_sores = cross_val_score(m, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


if __name__ == '__main__':
    cv_score()
    # cv()
    # run()
