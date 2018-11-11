import feature_selection
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from src import data_analyzing


def get_importance_plot():
    X, y = data_analyzing.get_data()

    # X = X.drop(selected, axis=1)
    # X = X.drop(maybe_dropped, axis=1)
    # X = X.drop(dropped, axis=1)
    X = X.drop(feature_selection.macro_dropped, axis=1)

    # Build a forest and compute the feature importances
    forest = ExtraTreesRegressor(n_estimators=250, random_state=0, n_jobs=-1)
    forest.fit(X, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    colors = ["r" if X.columns[i] in feature_selection.selected else "g" for i in indices]

    for f in range(X.shape[1]):
        print("%d. feature %d (%s, color: %s) : %f" % (
            f + 1, indices[f], X.columns[indices[f]], colors[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color=colors, align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


if __name__ == '__main__':
    get_importance_plot()
