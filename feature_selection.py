import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor

import data_analyzing

selected = [
    "cafe_count_5000_price_high",
    "cafe_count_3000",
    "cafe_avg_price_3000",
    "ekder_male"
]

# Build a classification task using 3 informative features
X, y = data_analyzing.get_data()

features = [
    'raion_build_count_with_material_info', 'build_count_block', 'build_count_wood', 'build_count_frame',
    'build_count_brick', 'build_count_monolith', 'build_count_panel', 'build_count_foam', 'build_count_slag',
    'build_count_mix', 'raion_build_count_with_builddate_info', 'build_count_before_1920', 'build_count_1921-1945',
    'build_count_1946-1970', 'build_count_1971-1995', 'build_count_after_1995', ]
X = X[features]

# Build a forest and compute the feature importances
forest = ExtraTreesRegressor(n_estimators=250, random_state=0)
forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%s) : %f" % (f + 1, indices[f], features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
