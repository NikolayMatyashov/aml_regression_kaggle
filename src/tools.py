from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data_analyzing

X_train, X_test, y_train, y_test = data_analyzing.get_train_data(normalise=True, with_test_split=True)

X, y = data_analyzing.get_train_data(normalise=True)

models = {
    'xgboost': XGBRegressor(),
    'catboost': CatBoostRegressor(),
    'decision_tree': DecisionTreeRegressor()
}

parameters_for_grid = {
    'xgboost': {'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.07, 0.09],
                'gamma': [0],
                'subsample': [0.5, 0.75, 0.9],
                'colsample_bytree': [1],
                'max_depth': [7, 8, 9, 10]},
    'catboost': {'learning_rate': [0.25, 0.5, 0.75],
                 'max_depth': [5, 6, 7, 8]},
    'decision_tree': {'max_depth': [4, 5, 6, 7, 8, 9, 10]}
}

optimal_parameters = {
    'xgboost': {'n_estimators': [00],
                'learning_rate': [0.05],
                'gamma': [0],
                'subsample': [0.75],
                'colsample_bytree': [1],
                'max_depth': [8]},
    'catboost': {'learning_rate': [0.25],
                 'max_depth': [6]},
    'decision_tree': {'max_depth': [6]}
}


def single_run(model):
    """
    Performs single run of the given model (build, fit, predict and calculate mean error).
    Prints Mean Absolute Error for predicted Y.
    :param model: given model
    """
    global X_train, X_test, y_train, y_test

    model.fit(X_train, y_train)
    Y_hat = model.predict(X_test)
    MAE = np.mean(abs(Y_hat - y_test))
    print('MAE for given model : %.3f' % MAE)


def grid_search(model, tuned_params_dict, print_results=True, results_to_csv=False):
    """
    Performs grid search cross validation to find optimal parameters for the given model.
    Prints cross validation results.
    :param model: given model
    :param tuned_params_dict: dictionary of parameters to check
    :param print_results: if True - prints results to console (default: True)
    :param results_to_csv: if True - saves results into '../data/model_analysis.csv' (default: False)
    """
    global X, y

    # Use Grid Search
    tuned_parameters = [tuned_params_dict]
    clf = GridSearchCV(model, tuned_parameters, cv=5, scoring='neg_mean_absolute_error', n_jobs=4)
    clf.fit(X, y)

    # Assign results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    parameters = clf.cv_results_['params']

    # Print results if needed
    if print_results:
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # Save into csv if needed
    if results_to_csv:
        model_csv = pd.DataFrame(data=parameters)
        model_csv.insert(loc=0, column='STD', value=stds)
        model_csv.insert(loc=0, column='MAE', value=means)
        model_csv.to_csv(r'../data/model_analysis.csv')


def cv_score(model):
    """
    Performs cross validation on the chosen model and prints mean error and error for each iteration.
    """
    global X, y

    cros_val_sores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=4)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


def plot_parameter_distribution(model, chosen_param, tuned_parameters_list):
    """
    Builds bar chart demonstrating how chosen parameter of the model
    influences on error
    :param model: given model
    :param chosen_param: string name of the parameter (e.g. 'max_depth')
    :param tuned_parameters_list: array of possible values of the parameters (e.g. [6, 7, 8, 9])
    """
    global X, y

    tuned_parameters = [{chosen_param: tuned_parameters_list}]
    clf = GridSearchCV(model, tuned_parameters, cv=5, scoring='neg_mean_absolute_error', n_jobs=4)
    clf.fit(X, y)
    means = clf.cv_results_['mean_test_score']
    means = abs(means)
    stds = clf.cv_results_['std_test_score']

    # Build the plot
    x_pos = np.arange(len(tuned_parameters_list))
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=stds, color='r', align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('MAE')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tuned_parameters_list)
    ax.set_title(chosen_param)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ---- Uncomment needed tool ----

    # single_run(models['xgboost'])

    # grid_search(models['xgboost'], parameters_for_grid['xgboost'])

    # cv_score(models['xgboost'])

    # plot_parameter_distribution(list(parameters_for_grid['xgboost'].keys())[0],
    #                             parameters_for_grid['xgboost']['n_estimators'])

    pass
