from src import filtered_features
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from src import data_analyzing


def make_importance_plot(X, y, title="Features importance"):
    # Build a forest and compute the feature importances
    forest = ExtraTreesRegressor(n_estimators=250, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Calculate std
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # Print the feature ranking
    print("Feature ranking:")
    for i in range(X.shape[1]):
        print("%d. %s -  importance: %f, std: %f" % (i + 1, X.columns[indices[i]], importances[indices[i]], std[i]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.title(title)
    plt.show()

    plt.figure()
    plt.bar(range(X.shape[1]), importances[indices], align="center", yerr=std)
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.title(title + " with std")
    plt.show()

if __name__ == '__main__':
    X, y = data_analyzing.get_train_data(only_important_features=False)
    make_importance_plot(X, y, "Importance of all features")

    X, y = data_analyzing.get_train_data(only_important_features=True)
    make_importance_plot(X, y, "Importance of selected features")
