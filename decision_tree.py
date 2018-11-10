from sklearn.model_selection import GridSearchCV

import data_analyzing
from sklearn.tree import DecisionTreeRegressor
import numpy as np

N = 8

X_train, X_test, y_train, y_test = data_analyzing.get_normalised_data()


def run():
    m = DecisionTreeRegressor(max_depth=N)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for Decision tree : %.3f' % MAE)


def cv():
    tuned_parameters = [{'max_depth': [4, 5, 6, 7, 8, 9, 10]}]

    score = 'neg_mean_absolute_error'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5,
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
    print('MAE for Decision tree : %.3f' % MAE)
    print()


if __name__ == '__main__':
    # cv()
    run()
